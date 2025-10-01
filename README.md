# Office AI Agent System

본 저장소는 문서 임베딩, 하이브리드 검색(RAG), FastAPI 백엔드, Streamlit 대시보드 등으로 구성된 사내 문서 분석/질의 응답 시스템입니다.

## 주요 구성 요소
- RAG 파이프라인: 임베딩(`sentence-transformers`), 벡터 검색(`faiss-cpu`), BM25(`rank-bm25`), 생성(`transformers`)
- 백엔드: `FastAPI` + `uvicorn`
- 분석/대시보드: `Streamlit`, `matplotlib`, `seaborn`, `scikit-learn`
- 문서 처리: `PyMuPDF(pymupdf)`, `pdfplumber`, `python-docx`, `beautifulsoup4`

## 요구 사항 설치
Python 3.10+ 권장.

```bash
# 가상환경 생성 및 활성화 (예시)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -U pip
pip install -r requirements.txt
```

## 사용 방법
### 1) 백엔드 서버 실행 (FastAPI)
```bash
python backend.py
# 또는
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
- API 문서: `http://localhost:8000/docs`

### 2) Streamlit 대시보드 (제조 분석 예시)
```bash
cd Manufacturing_Analysis
streamlit run dashboard.py
```

### 3) RAG 파이프라인 스크립트 예시
```bash
python rag_agent_main.py
```

## 프로젝트 구조 (요약)
```
Office_AI_Agent_System/
├─ app.py
├─ backend.py
├─ rag_agent_main.py
├─ Manufacturing_Analysis/
│  ├─ dashboard.py
│  └─ requirements.txt
├─ parsing/, chunking/, temp/rag/ ...
├─ requirements.txt
└─ README.md
```

## 추가 참고
- GPU 사용 시 `torch` CUDA 빌드를 별도 설치해야 할 수 있습니다.
- 한국어 형태소 분석을 위해 `konlpy` 사용 시 Java 설치가 필요할 수 있습니다.
- PDF, 워드, 웹 문서 등 다양한 소스에서 텍스트를 추출합니다.

문의/개선 제안은 이슈로 남겨주세요.
