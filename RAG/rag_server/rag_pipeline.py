"""
위니쿠키 RAG 파이프라인 모듈

ChromaDB 검색 → 임계값 필터링 → 프롬프트 조합 → Bllossom-3B 답변 생성
4개 모듈을 통합한 RAG 파이프라인입니다.
"""

import json
import chromadb
import requests
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


# ============================================================
# 설정값
# ============================================================
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "cookie-chatbot-bllossom"
CHROMA_DB_PATH = "../chroma_db"
COLLECTION_NAME = "winicookie_knowledge"
DISTANCE_THRESHOLD = 0.45


# ============================================================
# 임베딩 함수
# ============================================================
class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Ollama 로컬 서버의 임베딩 모델을 ChromaDB에서 사용하기 위한 커스텀 함수

    - model: Ollama에 설치된 임베딩 모델 이름
    - url: Ollama 서버 주소 (기본: localhost:11434)
    """

    def __init__(self, model: str = EMBEDDING_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = requests.post(
                f"{self.url}/api/embed", json={"model": self.model, "input": text}
            )
            embeddings.append(response.json()["embeddings"][0])
        return embeddings


# ============================================================
# ChromaDB 초기화
# ============================================================
def init_chromadb(db_path: str = CHROMA_DB_PATH) -> chromadb.Collection:
    """ChromaDB 클라이언트와 컬렉션을 초기화"""
    ollama_ef = OllamaEmbeddingFunction()
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# 모듈 로드 시 컬렉션 초기화
collection = init_chromadb()


# ============================================================
# 모듈 1: Knowledge Base 검색
# ============================================================
def search_knowledge(query: str, n_results: int = 3) -> list:
    """
    사용자 질문으로 ChromaDB에서 관련 문서를 검색

    Args:
        query: 사용자 질문 텍스트
        n_results: 검색할 문서 수 (기본값: 3)

    Returns:
        list of dict: [{'content': str, 'category': str, 'distance': float}, ...]
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    search_results = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        search_results.append(
            {"content": doc, "category": meta["category"], "distance": dist}
        )

    return search_results


# ============================================================
# 모듈 2: 임계값 판단 (환각 방지 1차 필터)
# ============================================================
def filter_by_threshold(
    search_results: list, threshold: float = DISTANCE_THRESHOLD
) -> dict:
    """
    검색 결과의 유사도 거리를 기준으로 관련 정보 존재 여부를 판단

    Args:
        search_results: search_knowledge()의 반환값
        threshold: 유사도 거리 임계값 (기본값: 0.45)

    Returns:
        dict: {
            'has_answer': bool,
            'filtered_results': list,
            'min_distance': float
        }
    """
    filtered = [r for r in search_results if r["distance"] <= threshold]
    min_distance = min(r["distance"] for r in search_results)

    return {
        "has_answer": len(filtered) > 0,
        "filtered_results": filtered,
        "min_distance": min_distance,
    }


# ============================================================
# 모듈 3: 프롬프트 조합 (환각 방지 2차 필터)
# ============================================================
def build_prompt(query: str, filtered_results: list) -> str:
    """
    검색 결과와 사용자 질문을 결합하여 LLM 프롬프트를 생성

    Args:
        query: 사용자 질문
        filtered_results: 임계값을 통과한 검색 결과 리스트

    Returns:
        str: Bllossom-3B에 전달할 완성된 프롬프트
    """
    context_lines = []
    for i, r in enumerate(filtered_results, 1):
        context_lines.append(f"{i}. [{r['category']}] {r['content']}")
    context_text = "\n".join(context_lines)

    prompt = f"""너는 '위니쿠키' 쿠키 전문점의 친절한 안내 챗봇이야.
아래 [참고 정보]만을 근거로 고객의 질문에 답변해.

중요한 규칙:
- 참고 정보에 있는 내용은 정확히 답변할 것
- 참고 정보에 없는 내용은 절대 추측하거나 만들어내지 말 것
- 질문에 여러 항목이 있으면, 참고 정보에 있는 항목은 답변하고 없는 항목은 "해당 정보가 없습니다"라고 각각 구분하여 답할 것
- 모든 항목이 참고 정보에 없으면 "죄송합니다. 010-3387-5313으로 문의해 주세요."라고 답할 것.
- 짧고 친절하게 답변할 것

[참고 정보]
{context_text}

[고객 질문]
{query}

[답변]"""

    return prompt


# ============================================================
# 모듈 4: Bllossom-3B 답변 생성
# ============================================================
def generate_answer(prompt: str) -> str:
    """
    Ollama API를 통해 Bllossom-3B 모델로 답변 생성

    Args:
        prompt: build_prompt()에서 생성된 프롬프트

    Returns:
        str: 모델이 생성한 답변 텍스트
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 256},
        },
    )

    return response.json()["response"].strip()


# ============================================================
# 통합: RAG 파이프라인
# ============================================================
NO_ANSWER_MESSAGE = (
    "죄송합니다, 해당 정보를 찾을 수 없습니다. 010-3387-5313으로 문의해 주세요."
)


def rag_pipeline(query: str, verbose: bool = False) -> dict:
    """
    RAG 파이프라인 통합 함수

    사용자 질문 → 검색 → 임계값 판단 → 프롬프트 조합 → 답변 생성

    Args:
        query: 사용자 질문
        verbose: True이면 중간 과정 정보를 반환에 포함

    Returns:
        dict: {
            'answer': str,
            'has_answer': bool,
            'min_distance': float,
            'num_sources': int,
            'sources': list (verbose일 때만)
        }
    """
    # 모듈 1: ChromaDB 검색
    search_results = search_knowledge(query)

    # 모듈 2: 임계값 판단
    filter_result = filter_by_threshold(search_results)

    # 임계값 초과 → 정보 없음 응답 (1차 필터)
    if not filter_result["has_answer"]:
        result = {
            "answer": NO_ANSWER_MESSAGE,
            "has_answer": False,
            "min_distance": filter_result["min_distance"],
            "num_sources": 0,
        }
        if verbose:
            result["sources"] = search_results
        return result

    # 모듈 3: 프롬프트 조합
    prompt = build_prompt(query, filter_result["filtered_results"])

    # 모듈 4: 답변 생성
    answer = generate_answer(prompt)

    result = {
        "answer": answer,
        "has_answer": True,
        "min_distance": filter_result["min_distance"],
        "num_sources": len(filter_result["filtered_results"]),
    }
    if verbose:
        result["sources"] = filter_result["filtered_results"]

    return result
