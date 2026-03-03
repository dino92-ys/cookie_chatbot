# 🍪 위니쿠키 AI 챗봇 — Fine-tuning + RAG 파이프라인

> 소규모 도메인 특화 챗봇을 **로컬 환경에서** 파인튜닝하고, RAG로 환각(Hallucination)을 제어한 엔드투엔드 프로젝트

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-Apple_Silicon-black)](https://github.com/ml-explore/mlx)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-green)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688)](https://fastapi.tiangolo.com)

---

## 프로젝트 개요

실제 운영 중인 수제 쿠키 전문점 **위니쿠키**의 고객 응대 챗봇을 구축한 프로젝트입니다.

**핵심 과제**: 3B 소형 모델로 메뉴, 가격, 영업시간 등 사실 정보를 정확하게 답변하되, 없는 정보를 만들어내는 환각(Hallucination) 문제를 해결해야 했습니다.

**접근 방식**: Fine-tuning 단독으로는 환각을 제어할 수 없다는 것을 실험으로 확인한 후, RAG 파이프라인을 도입하여 이중 안전장치(유사도 임계값 필터 + 프롬프트 지시)로 환각을 차단했습니다.

### 주요 성과

| 지표 | Fine-tuning Only | + RAG 적용 후 |
|------|:-:|:-:|
| 사실 질문 정확도 | ✅ 정확 | ✅ 정확 |
| 존재하지 않는 메뉴 질문 | ❌ 환각 발생 ("마카롱도 팔아요!") | ✅ "해당 정보를 찾을 수 없습니다" |
| 범위 밖 질문 (레시피, 사장님 이름 등) | ❌ 환각 발생 | ✅ 정보 없음 안내 |
| 환각 테스트 통과율 (10개 케이스) | — | 7 PASS / 1 PARTIAL / 2 FAIL → 고도화 후 개선 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 질문 입력                        │
└─────────────┬───────────────────────────────────────────┘
              ▼
┌─────────────────────────┐
│  FastAPI 서버 (app.py)   │
└─────────────┬───────────┘
              ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG 파이프라인                            │
│                                                           │
│  ① 질문 임베딩 (bge-m3)                                  │
│           ▼                                               │
│  ② ChromaDB 유사도 검색 (코사인, Top-3)                   │
│           ▼                                               │
│  ③ 임계값 필터 (distance ≤ 0.45 → 통과)  ← 환각 방지 1차 │
│           ▼                                               │
│  ④ 프롬프트 조합 (검색 결과 + 질문 + 규칙)  ← 환각 방지 2차│
│           ▼                                               │
│  ⑤ Bllossom-3B (Fine-tuned) 답변 생성                    │
└─────────────────────────────────────────────────────────┘
              ▼
┌─────────────────────────┐
│      답변 반환 (JSON)     │
└─────────────────────────┘
```

---

## 기술 스택

| 구분 | 기술 | 선택 이유 |
|------|------|----------|
| **LLM** | Bllossom-3B (Llama 3.2 기반) | 한국어 사전학습 150GB, Ollama 네이티브 지원 |
| **Fine-tuning** | MLX LoRA | Apple Silicon 최적화, 메모리 효율적 학습 |
| **임베딩** | bge-m3 | 다국어(Multi-lingual) 지원, 한국어 임베딩 품질 우수 |
| **벡터 DB** | ChromaDB | 경량 오픈소스, Python 친화적, 로컬 실행 |
| **LLM 서빙** | Ollama | 로컬 LLM 서빙, GGUF 모델 관리 |
| **API 서버** | FastAPI | 비동기 지원, 자동 API 문서 생성 |
| **개발 환경** | MacBook Pro 14 (M4 Pro, 48GB) | 로컬에서 학습~배포까지 완결 |

---

## 프로젝트 구조

```
cookie_llm/
├── README.md
├── lora_config.yaml                    # LoRA 학습 하이퍼파라미터 설정
│
├── data/                               # Fine-tuning 학습 데이터
│   ├── train.jsonl                     #   학습 데이터 (80개 대화 쌍)
│   ├── valid.jsonl                     #   검증 데이터
│   └── test.jsonl                      #   테스트 데이터
│
├── adapters-bllossom/                  # LoRA 어댑터 (학습 결과물)
│   ├── adapter_config.json
│   └── adapters.safetensors
│
├── RAG/
│   ├── winicookie_knowledge.json       #   Knowledge Base (46개 사실 문서)
│   ├── winicookie_embedding.ipynb      #   임베딩 + RAG 파이프라인 개발 노트북
│   ├── chroma_db/                      #   ChromaDB 벡터 저장소 (자동 생성)
│   │
│   └── rag_server/                     # API 서버
│       ├── app.py                      #   FastAPI 엔드포인트
│       └── rag_pipeline.py             #   RAG 파이프라인 모듈
│
└── 진행사항 정리/                       # 학습 기록 문서
    └── 쿠키챗봇_Bllossom3B_파인튜닝_배포_정리.md
```

---

## 진행 과정

### Phase 1: 모델 선정 및 Fine-tuning

**1-1. 모델 선정 과정**

처음에는 Qwen3-4B-Instruct로 시작했으나, Ollama 배포 단계에서 아키텍처 미지원 문제가 발생했습니다. GGUF 변환 시 양자화 관련 에러와 품질 손실도 확인되었습니다.

이를 통해 **배포 플랫폼의 호환성을 먼저 확인하고 모델을 선택해야 한다**는 교훈을 얻었고, Llama 3.2 기반의 Bllossom-3B로 전환했습니다.

| | Qwen3-4B | Bllossom-3B |
|---|:-:|:-:|
| Ollama 지원 | ❌ 아키텍처 미지원 | ✅ 네이티브 지원 |
| GGUF 변환 | ⚠️ 복잡, 품질 손실 | ✅ 깔끔한 변환 |
| 한국어 | 보통 | 우수 (150GB 사전학습) |

**1-2. LoRA Fine-tuning**

```yaml
# lora_config.yaml 주요 설정
iters: 250
batch_size: 4
num_layers: 16
learning_rate: 1e-5
mask_prompt: true    # assistant 응답만 학습
```

80개의 위니쿠키 Q&A 대화 데이터로 학습하여, 메뉴/가격/영업시간/주문방법 등 도메인 지식을 모델에 주입했습니다. `mask_prompt` 옵션으로 system/user 메시지는 loss에서 제외하고 assistant 응답만 효율적으로 학습했습니다.

**1-3. Ollama 배포**

MLX 퓨징 → GGUF 변환(f16) → Ollama Modelfile 작성 → 로컬 서빙 완료. 3B 모델이므로 f16 정밀도를 유지하면서도 약 6GB로 가볍게 구동됩니다.

### Phase 2: RAG 파이프라인 구축 및 환각 제어

**2-1. 문제 발견: Fine-tuning만으로는 부족하다**

Fine-tuning 된 모델은 학습된 패턴에 대해서는 정확하지만, 존재하지 않는 메뉴나 오타 입력에 대해 그럴듯한 거짓 정보를 생성하는 환각 문제가 발생했습니다.

```
Q: "마카롱 있어요?"  →  A: "네, 마카롱도 판매합니다!"  (❌ 환각)
Q: "버터쿠니 얼마예요?"  →  A: "3,500원입니다."  (❌ 환각, 실제 버터쿠키는 2,200원)
```

**2-2. Knowledge Base 설계**

RAG용 Knowledge Base는 Fine-tuning 데이터와 근본적으로 다릅니다. 질문-답변 쌍이 아닌 **개별 사실 문서** 단위로 설계하여, 검색 시 정확한 하나의 사실이 매칭되도록 했습니다.

```json
{
    "id": "menu_choco_fudge",
    "category": "메뉴",
    "content": "초코퍼지 가격은 3,500원입니다."
}
```

6개 카테고리(가게정보, 위치, 영업시간, 주문, 메뉴, 추천)에 걸쳐 46개의 사실 문서를 구축했습니다.

**2-3. 이중 환각 방지 메커니즘**

- **1차 필터 (유사도 임계값)**: ChromaDB 검색 결과의 코사인 거리가 0.45를 초과하면 답변 생성을 차단하고 "정보 없음"으로 응답합니다. 정답이 있는 질문(거리 ~0.17)과 없는 질문(거리 ~0.49)의 확연한 차이를 실험으로 확인한 뒤 설정한 값입니다.
- **2차 필터 (프롬프트 지시)**: 모델에게 "참고 정보에 없는 내용은 절대 추측하지 말 것"이라는 명시적 지시를 포함하여, 임계값을 통과했더라도 모델 레벨에서 환각을 억제합니다.

### Phase 3: RAG 품질 고도화

초기 환각 테스트(10개 케이스)에서 발견된 문제를 개선했습니다.

- **복합 질문 처리 개선**: "초코퍼지랑 마카롱 가격 알려줘"처럼 존재/비존재 정보가 혼합된 질문에서 아는 것은 답하고 모르는 것은 구분하여 안내하도록 프롬프트를 개선했습니다.
- **임계값 재튜닝**: 다양한 질문 유형에 대한 유사도 거리 분포를 분석하여 임계값을 0.50에서 0.45로 조정했습니다.
- **답변 톤/품질 개선**: 쿠키 전문점에 적합한 친절하고 간결한 답변 톤을 프롬프트 엔지니어링으로 보정했습니다.
- **Knowledge Base 보강**: 3B 모델이 추론하기 어려운 암묵적 정보(일요일 휴무, 음료 미판매 등)를 명시적 부정 문서로 추가하여 환각을 사전 차단했습니다.

---

## 실행 방법

### 사전 요구사항

- [Ollama](https://ollama.ai) 설치 및 실행
- Ollama에 `bge-m3` 임베딩 모델 설치: `ollama pull bge-m3`
- Ollama에 `cookie-chatbot-bllossom` 모델 등록 (Fine-tuned GGUF)
- Python 3.11+ / conda 환경 권장

### 1. 의존성 설치

```bash
conda activate ai_study_env

pip install chromadb fastapi uvicorn requests
```

### 2. Knowledge Base 임베딩 (최초 1회)

```bash
cd RAG/
jupyter notebook winicookie_embedding.ipynb
# → 셀 순서대로 실행하면 chroma_db/ 디렉토리가 자동 생성됩니다
```

### 3. API 서버 실행

```bash
cd RAG/rag_server/
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. API 테스트

```bash
# 기본 질문
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "초코퍼지 얼마예요?"}'

# 상세 응답 (검색 소스 포함)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "마카롱 있어요?", "verbose": true}'

# 헬스체크
curl http://localhost:8000/health
```

### API 응답 예시

```json
{
  "answer": "초코퍼지는 3,500원입니다!",
  "has_answer": true,
  "min_distance": 0.1766,
  "num_sources": 1
}
```

---

## 핵심 교훈

1. **Fine-tuning ≠ 사실 정확성**: 소형 모델의 파인튜닝은 대화 패턴을 학습할 뿐, 사실 관계의 정확성을 보장하지 않습니다. 도메인 특화 사실 정보에는 RAG가 필수입니다.

2. **배포 호환성 우선 확인**: 모델 선택 시 학습 성능뿐 아니라 배포 플랫폼(Ollama 등)과의 호환성을 먼저 확인해야 합니다. Qwen3 → Bllossom 전환으로 이를 체감했습니다.

3. **명시적 부정 정보의 중요성**: 3B 모델은 "월~토 영업" → "일요일 휴무"와 같은 암묵적 추론이 불안정합니다. Knowledge Base에 부정 정보를 명시적으로 포함시키는 것이 환각 방지에 효과적입니다.

4. **이중 안전장치 설계**: 단일 필터에 의존하지 않고, 유사도 임계값(검색 단계)과 프롬프트 지시(생성 단계) 양쪽에서 환각을 차단하는 것이 안정적입니다.

---

## 개발 환경

- **하드웨어**: MacBook Pro 14 (Apple M4 Pro, 48GB RAM)
- **OS**: macOS
- **패키지 관리**: conda (miniforge), `ai_study_env` 환경
- **주요 라이브러리**: mlx-lm, chromadb, fastapi, requests
- **로컬 LLM 서빙**: Ollama
- **벡터 변환**: llama.cpp (GGUF 변환)


