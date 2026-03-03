# 🍪 위니쿠키 AI 챗봇 — RAG 파이프라인 + OpenClaw 연동

로컬 LLM(Bllossom-3B)과 RAG(Retrieval-Augmented Generation)를 활용한 쿠키 전문점 고객 응대 챗봇입니다.

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목적** | 여자친구의 쿠키 가게(위니쿠키)를 위한 자동 고객 응대 챗봇 |
| **핵심 기술** | RAG (ChromaDB + bge-m3) + Fine-tuned Bllossom-3B |
| **실행 환경** | MacBook Pro M4 Pro (48GB RAM), 100% 로컬 실행 |
| **비용** | 무료 (클라우드 API 없음) |

## 아키텍처

```
고객 메시지
    │
    ▼
OpenClaw (채팅 채널: Telegram/WebChat)
    │
    ▼
FastAPI 서버 (localhost:8000)
    │
    ├─ [모듈1] ChromaDB 검색 (bge-m3 임베딩)
    │   └─ 39개 Knowledge Base 문서에서 유사 문서 Top-3 검색
    │
    ├─ [모듈2] 임계값 필터링 (환각 방지 1차)
    │   └─ cosine distance ≤ 0.45 기준으로 관련 문서만 통과
    │
    ├─ [모듈3] 프롬프트 조합 (환각 방지 2차)
    │   └─ 시스템 지시 + 검색 결과 + 질문을 결합한 제약 프롬프트
    │
    └─ [모듈4] Bllossom-3B 답변 생성 (Ollama)
        └─ temperature: 0.3 / top_p: 0.9 / max_tokens: 256
    │
    ▼
고객에게 답변 전달
```

## 핵심 설계 결정 및 근거

### 1. Fine-tuning만으로는 부족한 이유 → RAG 도입

초기에 Bllossom-3B를 위니쿠키 데이터로 fine-tuning하여 챗봇을 구현했으나, **사실 정확도(factual accuracy) 문제**가 발생했습니다. Fine-tuning은 모델의 어조와 응답 스타일은 학습시킬 수 있지만, 구체적인 가격, 영업시간 등의 **사실 정보를 안정적으로 기억시키기 어렵습니다.**

이를 해결하기 위해 RAG를 도입하여, 모델이 응답 생성 시 반드시 Knowledge Base에서 검색된 정보만을 참조하도록 설계했습니다.

### 2. 이중 환각 방지 메커니즘

환각(hallucination)을 방지하기 위해 두 단계의 필터를 설계했습니다:

- **1차 필터 (거리 임계값)**: ChromaDB 검색 결과의 cosine distance가 0.45를 초과하면, LLM을 호출하지 않고 즉시 "정보 없음" 응답을 반환합니다. 이는 관련 없는 문서가 프롬프트에 포함되어 모델이 잘못된 추론을 하는 것을 원천 차단합니다.

- **2차 필터 (프롬프트 제약)**: 임계값을 통과한 경우에도, 프롬프트에 "참고 정보에 없는 내용은 절대 추측하지 말 것"이라는 명시적 지시를 포함하여, 모델이 검색 결과 외의 정보를 생성하지 않도록 합니다.

### 3. 임계값 0.45 설정 근거

테스트를 통해 확인한 거리 분포:
- 정답이 있는 질문 ("초코퍼지 가격"): distance ≈ 0.18~0.25
- 정답이 없는 질문 ("마카롱 있어요?"): distance ≈ 0.55~0.70

0.45는 이 두 분포 사이의 안전 마진을 확보한 값입니다.

## 환각 테스트 결과

5가지 유형, 10개 테스트 케이스로 검증했습니다:

| 유형 | 질문 예시 | 환각 위험도 | 결과 |
|------|-----------|-------------|------|
| 존재하지 않는 정보 | "마카롱 있어요?" | 중 | PASS |
| 복합 질문 (있는 것 + 없는 것) | "초코퍼지랑 마카롱 가격 알려줘" | **최고** | 추후 재검증 |
| 추론/비교 질문 | "가장 인기 있는 메뉴?" | 높 | PASS |
| 유사하지만 틀린 정보 | "강남점 주소 알려줘" | 높 | PASS |
| 범위 밖 질문 | "쿠키 레시피 알려줘" | 중 | PASS |

> 복합 질문은 가장 높은 환각 위험도를 보이며, 임계값 필터를 통과하면서도 일부 오답이 포함될 수 있습니다. 이는 Knowledge Base에 명시적 부정 정보를 추가하여 개선 예정입니다.

## 설치 및 실행

### 사전 요구사항

- Python 3.10+ (conda/miniforge 권장)
- Ollama 설치 및 실행 중
- `bge-m3` 임베딩 모델: `ollama pull bge-m3`
- `cookie-chatbot-bllossom` 모델: Ollama에 등록 완료

### 1단계: RAG API 서버 실행

```bash
cd cookie_llm/RAG/rag_server

# 패키지 설치 (conda 환경)
conda install fastapi uvicorn chromadb requests -y

# 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

서버 확인:
```bash
# 헬스체크
curl http://localhost:8000/health

# 채팅 테스트
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "초코퍼지 얼마예요?"}'
```

### 2단계: OpenClaw 연동 (선택)

```bash
# OpenClaw 설치
npm install -g openclaw@latest

# 온보딩
openclaw onboard --install-daemon

# 설정 파일 적용 (openclaw_config.json 참고)
# ~/.openclaw/openclaw.json에 Ollama 프로바이더 설정 추가
```

## 프로젝트 구조

```
cookie_llm/
├── data/                          # 학습 데이터
├── adapters-bllossom/             # Fine-tuning 어댑터
├── RAG/
│   ├── chroma_db/                 # ChromaDB 벡터 저장소
│   ├── winicookie_knowledge.json  # Knowledge Base (39개 문서)
│   ├── winicookie_embedding.ipynb # 원본 개발 노트북
│   └── rag_server/
│       ├── rag_pipeline.py        # RAG 파이프라인 모듈 (4개 모듈 통합)
│       ├── app.py                 # FastAPI 서버
│       └── openclaw_config.json   # OpenClaw 연동 설정
└── winicookie_train.jsonl         # Fine-tuning 학습 데이터
```

## 기술 스택

| 분류 | 기술 | 역할 |
|------|------|------|
| LLM | Bllossom-3B (LoRA fine-tuned) | 한국어 답변 생성 |
| 임베딩 | bge-m3 (Ollama) | 한국어 시맨틱 검색 |
| 벡터 DB | ChromaDB | Knowledge Base 저장 및 검색 |
| API | FastAPI + Uvicorn | REST API 서버 |
| 추론 엔진 | Ollama | 로컬 LLM 배포 |
| 에이전트 | OpenClaw | 채팅 채널 연동 |
| 실행 환경 | M4 Pro MacBook (48GB RAM) | 전체 로컬 실행 |

## 향후 개선 계획

1. **명시적 부정 정보 추가**: Knowledge Base에 "위니쿠키에 없는 메뉴" 등의 부정 문서를 추가하여 복합 질문의 환각 방지 강화
2. **임계값 재튜닝**: 더 많은 테스트 케이스로 최적 임계값 탐색
3. **OpenClaw 스킬 개발**: 주문 접수, FAQ 자동 응답 등 확장 기능
4. **응답 속도 최적화**: MLX 활용한 Apple Silicon 최적화 검토
