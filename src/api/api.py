import os
import sys
from typing import Any, Dict, List, Optional

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


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "SmartLogi RAG API is running.",
        "usage": "Send POST requests to /api/chat or /api/query with a JSON body containing 'query'."
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        llm_client = LLMClient()
        llm_ready = llm_client.health_check()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return {
        "status": "ok" if llm_ready else "degraded",
        "ready": llm_ready,
        "message": "SmartLogi RAG API is ready" if llm_ready else "LM Studio is not reachable"
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        pipeline = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    try:
        result = pipeline.run(
            query=request.query,
            faq_threshold=request.faq_threshold,
            top_k=request.top_k,
            temperature=request.temperature,
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
def query(request: ChatRequest) -> ChatResponse:
    return chat(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=False)
