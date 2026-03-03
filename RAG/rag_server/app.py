"""
위니쿠키 RAG API 서버

FastAPI 기반 REST API로 RAG 파이프라인을 제공합니다.
OpenClaw 또는 다른 클라이언트에서 이 API를 호출하여 챗봇 기능을 사용합니다.

실행 방법:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import rag_pipeline

app = FastAPI(
    title="위니쿠키 챗봇 API",
    description="ChromaDB + Bllossom-3B 기반 RAG 챗봇 API",
    version="1.0.0"
)


# ============================================================
# 요청/응답 모델
# ============================================================
class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str
    verbose: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "message": "초코퍼지 얼마예요?",
                "verbose": False
            }
        }


class ChatResponse(BaseModel):
    """채팅 응답"""
    answer: str
    has_answer: bool
    min_distance: float
    num_sources: int
    sources: list | None = None


# ============================================================
# API 엔드포인트
# ============================================================
@app.get("/")
def health_check():
    """서버 상태 확인"""
    return {"status": "running", "service": "위니쿠키 챗봇 API"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    채팅 엔드포인트

    사용자 메시지를 받아 RAG 파이프라인으로 답변을 생성합니다.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")

    try:
        result = rag_pipeline(request.message, verbose=request.verbose)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파이프라인 오류: {str(e)}")


@app.get("/health")
def detailed_health():
    """상세 헬스체크 - ChromaDB 연결 및 문서 수 확인"""
    try:
        from rag_pipeline import collection
        doc_count = collection.count()
        return {
            "status": "healthy",
            "chromadb_documents": doc_count,
            "model": "cookie-chatbot-bllossom"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
