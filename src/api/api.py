import json
import os
import sys
import asyncio
import logging
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.embeddings.embedder import Embedder
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from src.vectordb.database import VectorDBManager

app = FastAPI(title="SmartLogi RAG API", version="1.0.0")

logger = logging.getLogger("smartlogi.api")
logger.setLevel(logging.INFO)

# Readiness flag set after models are preloaded on startup
_pipeline_ready: bool = False


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


def _send_lark_reply(reply_text: str) -> bool:
    webhook_url = os.getenv("LARK_BOT_WEBHOOK_URL")
    if not webhook_url:
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
        print(f"[Lark] Failed to send reply: {exc}")
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
async def lark_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}

    if payload.get("challenge"):
        return {"challenge": payload.get("challenge")}

    message_text = _extract_lark_message_text(payload)
    if not message_text:
        return {"status": "ignored", "message": "No text message received"}

    global _pipeline_ready
    try:
        if not _pipeline_ready:
            await asyncio.to_thread(_get_pipeline)
            _pipeline_ready = True

        pipeline = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    try:
        result = await asyncio.to_thread(pipeline.run, message_text)
    except ConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG pipeline failed: {exc}") from exc

    reply_text = result.get("answer", "")
    # send reply in background so we can respond quickly to Lark
    reply_sent = False
    try:
        asyncio.create_task(asyncio.to_thread(_send_lark_reply, reply_text))
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
