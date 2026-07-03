import json
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from base.config import config
from base.logger import logger
from rag_qa.main import IntegratedQASystem

app = FastAPI(
    title="EduRAG",
    description="教育场景 RAG 问答 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

iqa = IntegratedQASystem()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(..., min_length=1, description="会话 ID")
    source_filter: Optional[str] = Field(None, description="学科过滤，如 java、ai")


class QueryResponse(BaseModel):
    answer: str
    session_id: str


def _validate_source_filter(source_filter: Optional[str]) -> None:
    if source_filter and source_filter not in config.VALID_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的学科来源: {source_filter}，允许值: {config.VALID_SOURCES}",
        )


def _stream_query(query: str, session_id: str, source_filter: Optional[str]):
    try:
        for token, done in iqa.query(query, source_filter=source_filter, session_id=session_id):
            payload = json.dumps({"token": token, "done": done}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
    except Exception as e:
        logger.error(f"查询失败: {e}")
        payload = json.dumps({"token": str(e), "done": True, "error": True}, ensure_ascii=False)
        yield f"data: {payload}\n\n"


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_info():
    return {
        "name": "EduRAG API",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/query",
            "query_stream": "POST /api/query/stream",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """非流式查询，等待完整答案后返回。"""
    _validate_source_filter(request.source_filter)

    collected = []
    try:
        for token, done in iqa.query(
            request.query,
            source_filter=request.source_filter,
            session_id=request.session_id,
        ):
            if token:
                collected.append(token)
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return QueryResponse(answer="".join(collected), session_id=request.session_id)


@app.post("/api/query/stream")
def query_stream(request: QueryRequest):
    """流式查询，以 SSE 格式逐 token 返回答案。"""
    _validate_source_filter(request.source_filter)

    return StreamingResponse(
        _stream_query(request.query, request.session_id, request.source_filter),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
