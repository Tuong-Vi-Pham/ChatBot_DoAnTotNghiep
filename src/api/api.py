import json
import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = Any
from starlette.middleware.base import BaseHTTPMiddleware

from database import SessionLocal, check_db_connection, get_db, init_db
from models.conversation_history import save_bot_reply, save_message

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.embeddings.embedder import Embedder
from src.agent.nodes.intent_router import IntentRouterNode
from src.intent_clarification.clarifier import IntentClarifier, reconstruct_query
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from src.vectordb.database import VectorDBManager

app = FastAPI(title="SmartLogi RAG API", version="1.0.0")
app.state.lark_token_manager = None

logger = logging.getLogger("smartlogi.api")
logger.setLevel(logging.INFO)

# Readiness flag set after models are preloaded on startup
_pipeline_ready: bool = False

# Deduplication cache for Lark webhook events
_processed_events = set()
_processed_events_lock = Lock()

# In-memory clarification sessions for Lark chat
# Format: {chat_id: {"options": [...], "original_query": str, "timestamp": float}}
_clarification_sessions: Dict[str, Dict[str, Any]] = {}
_clarification_sessions_lock = Lock()


def _load_env_file(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


from src.lark.auth import TenantAccessTokenManager


class LarkTenantTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = None
        manager = request.app.state.lark_token_manager
        if manager is not None:
            token = await asyncio.to_thread(manager.get_token)
        if token is None:
            token = os.getenv("LARK_TENANT_ACCESS_TOKEN")
        request.state.lark_tenant_access_token = token
        return await call_next(request)


@app.on_event("startup")
async def load_lark_configuration() -> None:
    _load_env_file()
    app_id = os.getenv("LARK_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET")
    if app_id and app_secret:
        app.state.lark_token_manager = TenantAccessTokenManager(app_id=app_id, app_secret=app_secret)
    else:
        app.state.lark_token_manager = None

    try:
        init_db()
    except Exception:
        logger.warning("Database initialization skipped because SQLAlchemy/MySQL dependencies are not available yet.", exc_info=True)


app.add_middleware(LarkTenantTokenMiddleware)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask the trained RAG assistant")
    faq_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    top_k: int = Field(default=4, ge=1, le=8)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    answer: str
    retrieved_from: str
    sources: List[Dict[str, Any]]
    status: str = "ok"


_pipeline: Optional[RAGPipeline] = None
_llm_client: Optional[LLMClient] = None


_clarifier: Optional[IntentClarifier] = None


def _get_clarifier() -> IntentClarifier:
    global _clarifier, _llm_client
    if _clarifier is not None:
        return _clarifier
    if _llm_client is None:
        _get_pipeline()
    _clarifier = IntentClarifier(llm_client=_llm_client)
    return _clarifier


def _requires_deterministic_agent_routing(query: str) -> bool:
    """Return whether an explicit priority, ticket, or approval request must reach the agent first.

    The Lark ambiguity layer is useful for vague questions, but it must not
    preempt a deterministic agent intent. The intent router
    remains the single source of truth for this decision.
    """
    intent = IntentRouterNode.execute({"user_query": query}).get("intent")
    return intent in {
        "PRIORITY_ANALYSIS",
        "TICKET_QUERY",
        "TICKET_ANALYSIS",
        "PRIORITY_APPROVAL_REQUEST",
        "PRIORITY_APPROVAL_DECISION",
        "PRIORITY_UPDATE_EXECUTION",
    }



def _get_pipeline() -> RAGPipeline:
    global _pipeline, _llm_client

    if _pipeline is not None:
        return _pipeline

    try:
        embedder = Embedder(model_name="BAAI/bge-m3")
        db_manager = VectorDBManager(embedder=embedder)
        retriever = HybridRetriever(db_manager=db_manager, embedder=embedder)
        llm_client = LLMClient()
        _llm_client = llm_client
        _pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)
        return _pipeline
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize SmartLogi RAG pipeline: {exc}") from exc


@app.on_event("startup")
async def preload_pipeline_on_startup() -> None:
    """Preload heavy models and the RAG pipeline at application startup.

    This avoids loading model weights on the first incoming request which
    causes unacceptable latency for external callers.
    """
    global _pipeline_ready
    try:
        # Run synchronous initialization in a thread to avoid blocking the loop
        await asyncio.to_thread(_get_pipeline)
        await asyncio.to_thread(_get_clarifier)
        _pipeline_ready = True
        logger.info("RAG pipeline preloaded successfully on startup")
    except Exception as e:
        _pipeline_ready = False
        logger.exception("Failed to preload RAG pipeline on startup: %s", e)


def _extract_lark_message_text(payload: Dict[str, Any]) -> Optional[str]:
    event = payload.get("event", payload)
    message = event.get("message") if isinstance(event, dict) else None

    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                text_value = parsed.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value.strip()
            if content.strip():
                return content.strip()

        if isinstance(message.get("text"), str) and message.get("text", "").strip():
            return message.get("text", "").strip()

    if isinstance(event, dict):
        for key in ("text", "content"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    text_value = parsed.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        return text_value.strip()
                return value.strip()

    return None


def _extract_lark_event_info(payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
    event = payload.get("event", payload)
    if not isinstance(event, dict):
        return {"text": None, "chat_id": None, "message_id": None, "sender_id": None, "sender_type": None}

    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    if not isinstance(message, dict):
        message = {}

    chat_id = (
        message.get("chat_id")
        or message.get("chatId")
        or event.get("chat_id")
        or event.get("chatId")
    )
    message_id = message.get("message_id") or message.get("messageId")
    sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
    sender_id = (
        sender.get("sender_id")
        or sender.get("id")
        or sender.get("user_id")
        or message.get("sender_id")
        or event.get("sender_id")
    )

    return {
        "text": _extract_lark_message_text(payload),
        "chat_id": chat_id if isinstance(chat_id, str) else None,
        "message_id": message_id if isinstance(message_id, str) else None,
        "sender_id": sender_id if isinstance(sender_id, str) else None,
        "sender_type": "user" if sender_id else "system",
    }


def _save_conversation_history(
    chat_id: Optional[str],
    message_id: Optional[str],
    content: Optional[str],
    sender_id: Optional[str] = None,
    sender_type: str = "user",
) -> Dict[str, Any]:
    if not chat_id or not message_id or not content:
        return {"status": "ignored", "message": "Missing chat_id, message_id, or content"}

    from models.conversation_history import ConversationHistory

    db = SessionLocal()
    try:
        existing = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.chat_id == chat_id)
            .filter(ConversationHistory.message_id == message_id)
            .first()
        )
        if existing is not None:
            return {
                "status": "exists",
                "chat_id": chat_id,
                "message_id": message_id,
                "id": existing.id,
            }

        record = ConversationHistory(
            chat_id=chat_id,
            message_id=message_id,
            sender_id=sender_id,
            sender_type=sender_type,
            content=content,
            event_type="message",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return {
            "status": "saved",
            "chat_id": chat_id,
            "message_id": message_id,
            "id": record.id,
        }
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to save Lark conversation history: %s", exc)
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@app.post("/api/lark/conversation-history")
def save_lark_conversation_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    chat_id = payload.get("chat_id")
    message_id = payload.get("message_id")
    content = payload.get("content")
    sender_id = payload.get("sender_id")
    sender_type = payload.get("sender_type", "user")

    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="content is required")

    if not chat_id or not message_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id and message_id are required")

    result = _save_conversation_history(
        chat_id=str(chat_id),
        message_id=str(message_id),
        content=content.strip(),
        sender_id=str(sender_id) if sender_id else None,
        sender_type=str(sender_type),
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message"))

    return result


def _send_lark_reply(reply_text: str, chat_id: Optional[str] = None, message_id: Optional[str] = None, tenant_access_token: Optional[str] = None) -> bool:
    if not tenant_access_token:
        tenant_access_token = os.getenv("LARK_TENANT_ACCESS_TOKEN")

    if not tenant_access_token:
        # Fallback to the bot webhook URL if server API auth is not configured.
        webhook_url = os.getenv("LARK_BOT_WEBHOOK_URL")
        if not webhook_url:
            print("[Lark] Missing LARK_TENANT_ACCESS_TOKEN and LARK_BOT_WEBHOOK_URL")
            return False

        try:
            response = requests.post(
                webhook_url,
                json={"msg_type": "text", "content": {"text": reply_text}},
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            print(f"[Lark] Failed to send reply via webhook fallback: {exc}")
            return False

    headers = {
        "Authorization": f"Bearer {tenant_access_token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    if message_id:
        url = f"https://open.larksuite.com/open-apis/im/v1/messages/{message_id}/reply"
        payload = {
            "content": json.dumps({"text": reply_text}),
            "msg_type": "text",
        }
        params = None
    elif chat_id:
        url = "https://open.larksuite.com/open-apis/im/v1/messages"
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": reply_text}),
        }
        params = {"receive_id_type": "chat_id"}
    else:
        print("[Lark] Missing chat_id and message_id for reply")
        return False

    try:
        response = requests.post(url, headers=headers, json=payload, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("code", 0) != 0:
            print(f"[Lark] API returned error: {result}")
            return False
        return True
    except Exception as exc:
        print(f"[Lark] Failed to send reply via server API: {exc}")
        return False


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "SmartLogi RAG API is running.",
        "usage": "Send POST requests to /api/chat or /api/query with a JSON body containing 'query'."
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Return overall health and readiness.

    `ready` indicates whether the RAG pipeline and models have been preloaded.
    """
    try:
        llm_ready = False
        # Check LM Studio reachability quickly
        try:
            llm = LLMClient()
            llm_ready = await asyncio.to_thread(llm.health_check)
        except Exception:
            llm_ready = False

        return {
            "status": "ok" if llm_ready and _pipeline_ready else ("degraded" if llm_ready or _pipeline_ready else "down"),
            "ready": _pipeline_ready,
            "lm_ready": llm_ready,
            "message": "SmartLogi RAG API is ready" if (_pipeline_ready and llm_ready) else "Initializing or degraded"
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.get("/test-db")
def test_db(db: Session = Depends(get_db)) -> Dict[str, str]:
    try:
        check_db_connection()
        return {"message": "MySQL connection successful"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {exc}",
        ) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    global _pipeline_ready
    if not _pipeline_ready:
        # Try to initialize lazily if startup preload failed
        try:
            await asyncio.to_thread(_get_pipeline)
            _pipeline_ready = True
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    try:
        pipeline = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    try:
        # Run the CPU / IO bound pipeline.run in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            pipeline.run,
            request.query,
            request.faq_threshold,
            request.top_k,
            request.temperature,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG pipeline failed: {exc}") from exc

    return ChatResponse(
        answer=result["answer"],
        retrieved_from=result["retrieved_from"],
        sources=result["sources"],
    )


@app.post("/api/query", response_model=ChatResponse)
async def query(request: ChatRequest) -> ChatResponse:
    return await chat(request)


@app.post("/api/lark/webhook")
@app.post("/lark/webhook")
async def lark_webhook(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}

    if payload.get("challenge"):
        return {"challenge": payload.get("challenge")}

    event_info = _extract_lark_event_info(payload)
    message_text = event_info["text"]
    if not message_text:
        return {"status": "ignored", "message": "No text message received"}

    chat_id = event_info.get("chat_id")
    message_id = event_info.get("message_id")
    header = payload.get("header", {})
    event_id = header.get("event_id") if isinstance(header, dict) else None
    dedup_key = event_id or message_id

    # 1. Deduplication check
    if dedup_key:
        with _processed_events_lock:
            if dedup_key in _processed_events:
                logger.info("Ignoring duplicate Lark webhook event: %s", dedup_key)
                return {"status": "ok", "message": "duplicate_ignored", "reply_sent": False}
            _processed_events.add(dedup_key)
            if len(_processed_events) > 5000:
                _processed_events.clear()

    if chat_id and message_id:
        save_message(
            chat_id=chat_id,
            message_id=message_id,
            content=message_text,
            sender_id=event_info.get("sender_id"),
            sender_type=event_info.get("sender_type", "user"),
            event_type="message",
        )

    global _pipeline_ready
    try:
        if not _pipeline_ready:
            await asyncio.to_thread(_get_pipeline)
            await asyncio.to_thread(_get_clarifier)
            _pipeline_ready = True

        pipeline = _get_pipeline()
        clarifier = _get_clarifier()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    # 2. Check pending Intent Clarification Session for this chat_id
    query_to_run: Optional[str] = None
    reply_text: Optional[str] = None

    session = None
    if chat_id:
        with _clarification_sessions_lock:
            session = _clarification_sessions.get(chat_id)

    if session:
        clean_text = message_text.strip().lower()

        if clean_text in {"cancel", "hủy", "huy", "reset", "clear", "x"}:
            with _clarification_sessions_lock:
                _clarification_sessions.pop(chat_id, None)
            reply_text = "Intent clarification reset. Please ask a new question."
        else:
            selected_idx = None
            options = session.get("options", [])

            # Match option 1, 2, 3 or option text
            if clean_text in {"1", "option 1", "opt 1", "1."} or clean_text.startswith("1."):
                selected_idx = 0
            elif clean_text in {"2", "option 2", "opt 2", "2."} or clean_text.startswith("2."):
                selected_idx = 1
            elif clean_text in {"3", "option 3", "opt 3", "3."} or clean_text.startswith("3."):
                selected_idx = 2
            else:
                # Check string match
                for idx, opt in enumerate(options):
                    if opt.lower() in clean_text or clean_text in opt.lower():
                        selected_idx = idx
                        break

            if selected_idx is not None and selected_idx < len(options):
                chosen_opt = options[selected_idx]
                selected_options = session.get("selected_options", [])
                if chosen_opt not in selected_options:
                    selected_options.append(chosen_opt)
                session["selected_options"] = selected_options

                # Reconstruct query using original_query + all selected intent options
                original_q = session.get("original_query", message_text)
                query_to_run = reconstruct_query(original_q, selected_options)
                logger.info(f"[Lark Webhook] Reconstructed query from selection: '{query_to_run}' (selected_options: {selected_options})")

                # Update timestamp but keep session alive for sequential choices (1 -> 2 -> 3)
                with _clarification_sessions_lock:
                    if chat_id in _clarification_sessions:
                        _clarification_sessions[chat_id]["timestamp"] = time.time()
            else:
                # User typed a brand new query while awaiting clarification, clear old session and evaluate fresh
                with _clarification_sessions_lock:
                    _clarification_sessions.pop(chat_id, None)

    # 3. If no session response, preserve deterministic priority routing before
    # consulting the LLM ambiguity layer.  Otherwise an LLM clarification can
    # intercept a valid priority-analysis request before the agent sees it.
    if reply_text is None and query_to_run is None:
        if _requires_deterministic_agent_routing(message_text):
            query_to_run = message_text
        else:
            try:
                clarify_result = await asyncio.to_thread(
                    clarifier.check_ambiguity,
                    message_text,
                    retriever=getattr(pipeline, "retriever", None)
                )
            except Exception as err:
                logger.warning("Ambiguity check failed: %s", err)
                clarify_result = {"is_ambiguous": False, "options": [], "original_query": message_text}

            if clarify_result.get("is_ambiguous") and len(clarify_result.get("options", [])) >= 1:
                opts = [opt.replace("*", "") for opt in clarify_result["options"]]
                if chat_id:
                    with _clarification_sessions_lock:
                        _clarification_sessions[chat_id] = {
                            "original_query": message_text,
                            "options": opts,
                            "selected_options": [],
                            "conversation_id": chat_id,
                            "relevant_context": clarify_result.get("relevant_context", ""),
                            "timestamp": time.time(),
                        }

                options_str = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(opts)])
                reply_text = (
                    f"Intent Clarification\n\n"
                    f"Your query \"{message_text.replace('*', '')}\" could refer to multiple topics. Please select an option by typing numbers (e.g. 1, 2, 3):\n\n"
                    f"{options_str}\n\n"
                    f"(Reply option number or type cancel to reset)"
                )
            else:
                query_to_run = message_text

    # 4. Run RAG Pipeline if we have a target query
    if query_to_run:
        try:
            result = await asyncio.to_thread(
                pipeline.run,
                query_to_run,
                chat_id=chat_id,
                current_message_id=message_id,
            )
            raw_answer = result.get("answer", "")
            reply_text = raw_answer
        except ConnectionError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG pipeline failed: {exc}") from exc

    if not reply_text:
        reply_text = "I could not generate a response. Please try again."

    # Strip all asterisks from reply text
    reply_text = reply_text.replace("*", "")

    bot_message_id = f"bot_{message_id}" if message_id else None
    if bot_message_id and chat_id:
        save_bot_reply(chat_id=chat_id, message_id=bot_message_id, reply_text=reply_text)

    # 5. Send reply via Lark API
    reply_sent = False
    try:
        asyncio.create_task(
            asyncio.to_thread(
                _send_lark_reply,
                reply_text,
                chat_id=event_info.get("chat_id"),
                message_id=event_info.get("message_id"),
                tenant_access_token=getattr(request.state, "lark_tenant_access_token", None),
            )
        )
        reply_sent = True
    except Exception:
        reply_sent = False

    return {
        "status": "ok",
        "reply_sent": reply_sent,
        "reply": reply_text,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=False)
