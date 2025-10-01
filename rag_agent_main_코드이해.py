#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG-based Internal Policy Agent
- Retrieval-Augmented Generation (RAG) 기술을 사용하여 사내 규정 문서를 기반으로 질문에 답변하는 시스템입니다.
- RAG는 두 단계로 작동합니다:
  1. Retrieval (검색): 사용자의 질문과 가장 관련성이 높은 문서를 대규모 문서 모음에서 찾아냅니다.
  2. Generation (생성): 검색된 문서를 '근거 자료'로 삼아, 대규모 언어 모델(LLM)이 질문에 대한 답변을 생성합니다.
- 이 방식은 LLM이 잘못된 정보를 생성(환각, Hallucination)하는 것을 방지하고, 특정 도메인(여기서는 사내 규정)에 대한 정확한 답변을 제공하도록 돕습니다.
- 이 코드의 목표는 최대 3개의 유사 조항을 검색하고, 선택적으로 LLM을 통해 요약 답변을 생성하는 것입니다.
"""

# ====================== 시스템 및 라이브러리 임포트 ======================
import os
# "TOKENIZERS_PARALLELISM" 환경 변수를 "false"로 설정합니다.
# Hugging Face의 Tokenizers 라이브러리가 여러 프로세스를 사용하여 병렬 처리를 시도할 때 발생할 수 있는 교착 상태(deadlock)를 방지합니다.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json  # JSON 파일(규정 데이터)을 읽고 쓰기 위해 사용합니다.
from typing import List, Dict, Any, Optional, Tuple  # 코드의 가독성과 안정성을 높이기 위한 타입 힌트 라이브러리입니다.
import re  # 정규 표현식을 사용하여 텍스트 패턴을 찾고 처리합니다. (예: 문서 포매팅)

import numpy as np  # 다차원 배열과 행렬 연산을 위한 라이브러리로, 임베딩 벡터를 처리하는 데 필수적입니다.
from sentence_transformers import SentenceTransformer  # 문장이나 단락을 의미적으로 유사한 벡터(임베딩)로 변환하는 모델을 쉽게 사용할 수 있게 해주는 라이브러리입니다.
import faiss  # Facebook AI Research에서 개발한 라이브러리로, 대규모 벡터 데이터셋에서 효율적으로 유사도 검색 및 클러스터링을 수행합니다. RAG의 '검색' 단계를 담당합니다.
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Hugging Face의 라이브러리로, LLM 모델과 토크나이저를 쉽게 로드하고 사용할 수 있게 해줍니다.
import torch  # PyTorch 라이브러리로, 딥러닝 모델(임베딩, LLM)을 실행하고 GPU 가속을 활용하기 위해 필요합니다.

# ====================== 사용자 설정 (Configuration) ======================
# 이 부분에서 시스템의 주요 파라미터를 쉽게 변경할 수 있습니다.

# 규정 데이터가 저장된 JSON 파일의 경로입니다.
# 데이터는 보통 [{"title": "규정 제목", "text": "규정 본문"}, ...] 형태로 구성됩니다.
CHUNKS_PATH = "/home/alpaco/kimcy/Office_AI_Agent_System/report/standard_flattened.json"

# 문장을 벡터로 변환(임베딩)할 때 사용할 모델의 이름입니다.
# 'jhgan/ko-sbert-nli'는 한국어 문장의 의미를 잘 포착하는 것으로 알려진 경량 모델입니다.
EMBED_MODEL_NAME = "jhgan/ko-sbert-nli"

# 답변 생성을 위해 사용할 대규모 언어 모델(LLM)의 이름입니다.
# 'skt/A.X-4.0-Light'는 SKT에서 개발한 한국어 LLM으로, 비교적 가벼워 빠른 응답 생성이 가능합니다.
LLM_MODEL_NAME = "skt/A.X-4.0-Light"

# LLM이 한 번에 처리할 수 있는 최대 토큰(단어 조각)의 수입니다. 이 길이를 넘으면 에러가 발생하므로, 프롬프트 길이를 관리해야 합니다.
MODEL_MAX_CONTEXT = 16384

# 사용자의 질문과 관련하여 FAISS에서 검색할 초기 후보 문서의 개수입니다.
# 이 값을 높이면 더 관련 있는 문서를 찾을 확률이 높아지지만, 처리 속도는 약간 느려질 수 있습니다.
TOP_K_RETRIEVE = 10

# 검색된 문서가 사용자의 질문과 의미적으로 얼마나 유사해야 최종 결과에 포함될지를 결정하는 임계값입니다.
# 1에 가까울수록 매우 유사해야 하며, 이 값을 낮추면 더 넓은 범위의 문서가 포함됩니다.
DEFAULT_SCORE_THRESHOLD = 0.45
# =======================================================================


# ====[최종 수정] 전문가급 문서 포매팅 함수 구현====
def format_document_text(text: str) -> str:
    """
    사내 규정과 같은 정형화된 텍스트의 가독성을 높이기 위해 구조적으로 포매팅합니다.
    법률 또는 규정 문서는 '항(①)', '호(1.)' 등으로 구성되는 경우가 많아, 이를 시각적으로 구분해주면 사용자가 이해하기 쉽습니다.

    - ①, ② 등 '항'은 새로운 문단으로 구분하여 시작합니다.
    - 1., 2. 등 '호'는 들여쓰기를 적용하여 계층 구조를 명확히 합니다.
    - 불필요한 연속 공백이나 줄바꿈을 정리하여 깔끔하게 만듭니다.
    """
    if not text: # 입력 텍스트가 비어있으면 빈 문자열을 반환합니다.
        return ""

    # 1. 텍스트 초기 정규화: 앞뒤 공백을 제거하고, 여러 개의 공백을 하나로 합칩니다.
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 2. '항' (②부터 ⑩까지의 원 문자) 앞에 줄바꿈 두 번을 추가하여 문단을 나눕니다.
    # '①'은 보통 조항의 시작이므로 제외하고, 두 번째 항부터 줄바꿈을 적용합니다.
    text = re.sub(r'\s*([②-⑩])\s*', r'\n\n\1 ', text)

    # 3. '호' (숫자와 점으로 시작하는 패턴) 앞에 줄바꿈과 들여쓰기(공백 4칸)를 추가합니다.
    text = re.sub(r'\s*(\d+\.)\s*', r'\n    \1 ', text)

    # 4. 일반 문장 끝 처리: 문장의 끝을 나타내는 '.' 뒤에 공백이 오면 줄바꿈을 추가하여 문장 구분을 명확히 합니다.
    # 단, '항'이나 '호'로 시작하는 줄은 이미 구조화되었으므로 이 규칙을 적용하지 않습니다.
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        stripped = line.strip()
        # '①'로 시작하거나 '1.' 같은 패턴으로 시작하지 않는 일반 문장인 경우에만
        if not (stripped.startswith('①') or re.match(r'^\d+\.', stripped)):
            line = line.replace('. ', '.\n') # '. '을 '.\n'으로 변경
        formatted_lines.append(line)

    return '\n'.join(formatted_lines) # 처리된 줄들을 다시 하나의 텍스트로 합쳐 반환합니다.


class RAGAgent:
    """
    RAG 파이프라인의 모든 로직을 캡슐화한 메인 클래스입니다.
    이 클래스의 인스턴스를 생성하면 규정 문서를 로드하고, 검색 시스템을 준비하며, 질문에 답변할 수 있는 상태가 됩니다.
    """

    def __init__(self,
                 chunks_path: str = CHUNKS_PATH,
                 embed_model_name: str = EMBED_MODEL_NAME,
                 llm_model_name: Optional[str] = LLM_MODEL_NAME,
                 use_gpu: bool = True):
        """
        RAGAgent 클래스의 생성자(초기화 메서드)입니다.
        - 필요한 모델과 데이터를 로드하고 검색 인덱스를 구축합니다.
        - use_gpu: GPU 사용 가능 여부를 확인하고 설정합니다.
        """
        self.chunks_path = chunks_path
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        # use_gpu 플래그가 True이고, 실제로 torch에서 사용 가능한 CUDA 장치가 있을 때만 GPU를 사용하도록 설정합니다.
        self.use_gpu = use_gpu and torch.cuda.is_available()

        print("임베딩 모델을 로딩합니다...")
        # 문장을 벡터로 변환하는 SentenceTransformer 모델을 로드합니다. 이 과정은 보통 수 초에서 수십 초가 소요됩니다.
        self.embed_model = SentenceTransformer(self.embed_model_name)

        # JSON 파일에서 규정 문서(chunks)와 키워드를 로드합니다.
        self.chunks, self.chunk_keywords = self._load_chunks_and_keywords(self.chunks_path)
        self.embeddings = None  # 문서 벡터들을 저장할 변수
        self.index = None       # FAISS 검색 인덱스를 저장할 변수
        self._build_index()     # 문서들을 벡터화하고 FAISS 인덱스를 구축하는 메서드를 호출합니다.

        # LLM 관련 변수들은 None으로 초기화합니다.
        # LLM은 메모리를 많이 차지하므로, 실제로 답변 '생성'이 필요할 때만 로드하는 '지연 로딩(Lazy Loading)' 방식을 사용합니다.
        self.generator = None
        self.tokenizer = None

        print("RAG 에이전트 준비 완료.")

    def _load_chunks_and_keywords(self, path: str) -> Tuple[List[Dict[str, str]], List[List[str]]]:
        """
        지정된 경로의 JSON 파일에서 규정 데이터를 로드하고, 검색 순위 조정을 위한 키워드를 추출합니다.
        - 다양한 형식의 JSON에 대응할 수 있도록 유연하게 키(key)를 탐색합니다.
        - 각 규정의 '제목'에서 키워드를 추출하여 나중에 키워드 기반 점수 보너스를 주는 데 사용합니다.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"규정 파일({path})을 찾을 수 없습니다.")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # JSON 데이터가 {"내용": [...]} 와 같은 래퍼(wrapper) 객체 안에 있을 경우를 처리합니다.
        if isinstance(data, dict) and "내용" in data:
            data = data["내용"]

        if not isinstance(data, list) or not data:
            raise ValueError("JSON 파일이 비어있거나 리스트 형식이 아닙니다.")

        chunks = []          # 포맷팅된 딕셔너리(title, text)를 저장할 리스트
        chunk_keywords = []  # 각 chunk의 제목에서 추출한 키워드 리스트를 저장

        first_item = data[0] # 첫 번째 데이터 항목을 기준으로 키 이름을 유추합니다.
        # 제목과 본문에 해당하는 키 이름이 다를 수 있으므로 가능한 후보 목록을 만듭니다.
        title_keys = ["title", "Title", "제목"]
        text_keys = ["text", "Text", "내용", "본문"]

        # 후보 목록 중 실제 데이터에 존재하는 키를 찾습니다.
        title_key = next((k for k in title_keys if k in first_item), None)
        text_key = next((k for k in text_keys if k in first_item), None)

        if not text_key: # 본문 키는 필수적이므로, 찾지 못하면 에러를 발생시킵니다.
            raise ValueError(f"JSON에서 본문 키를 찾을 수 없습니다. (시도한 키: {text_keys}, 실제 키: {list(first_item.keys())})")

        print(f"감지된 JSON 키 - 제목: '{title_key}', 본문: '{text_key}'")

        for i, item in enumerate(data):
            # title_key가 없으면 기본값으로 "chunk_1", "chunk_2" ... 와 같이 이름을 붙입니다.
            title = item.get(title_key, f"chunk_{i+1}") if title_key else f"chunk_{i+1}"
            text = item.get(text_key, "")

            if text.strip(): # 본문 내용이 있는 경우에만 처리합니다.
                chunks.append({"title": title, "text": text})

                # 제목에서 키워드를 추출하는 로직입니다.
                # "제1장 총칙", "제2조 (목적)"과 같은 형식적인 부분을 제거하여 핵심 단어만 남깁니다.
                clean_title = re.sub(r"제\d+장|제\d+조\s*", "", title).strip()
                # 제목에 포함된 특수문자(․, ·)를 공백으로 변경합니다.
                processed_title = clean_title.replace(" ․ ", " ").replace(" · ", " ")
                # 공백을 기준으로 단어를 분리하여 키워드 리스트를 만듭니다.
                keywords = processed_title.split()
                chunk_keywords.append([kw.strip() for kw in keywords if kw.strip()])

        print(f"{len(chunks)}개의 규정 조항(chunk)을 로드하고 키워드를 추출했습니다.")
        if not chunks:
            raise ValueError("유효한 내용이 있는 청크가 없습니다.")
        return chunks, chunk_keywords

    def _build_index(self):
        """
        로드된 규정 문서들(chunks)을 임베딩 벡터로 변환하고, FAISS 인덱스를 구축합니다.
        이 인덱스는 나중에 사용자 질문이 들어왔을 때 빠르고 효율적으로 유사한 문서를 찾는 데 사용됩니다.
        """
        if not self.chunks:
            self.index = None
            self.embeddings = None
            return
            
        texts = [c["text"] for c in self.chunks] # 모든 문서의 본문을 리스트로 만듭니다.
        print("규정 문서에 대한 임베딩 벡터를 생성합니다...")
        
        # SentenceTransformer 모델을 사용하여 모든 텍스트를 한 번에 벡터로 변환합니다.
        # normalize_embeddings=True: 벡터의 길이를 1로 정규화합니다. 이렇게 하면 내적(Inner Product)을 사용하여 코사인 유사도를 쉽게 계산할 수 있습니다.
        # show_progress_bar=True: 변환 진행 상황을 시각적으로 보여줍니다.
        embs = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embs = np.asarray(embs).astype("float32") # FAISS에서 사용하기 위해 numpy 배열과 float32 타입으로 변환합니다.
        self.embeddings = embs
        
        dim = embs.shape[1] # 임베딩 벡터의 차원 수를 가져옵니다. (예: 768)
        
        # FAISS 인덱스를 생성합니다.
        # IndexFlatIP: 'Flat'은 데이터를 압축하지 않고 그대로 사용한다는 의미이며, 'IP'(Inner Product)는 내적을 사용하여 벡터 간의 유사도를 계산하는 방식입니다.
        # 임베딩이 정규화되었기 때문에, 내적 값은 코사인 유사도와 동일하며, 1에 가까울수록 유사도가 높다는 의미입니다.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs) # 생성된 임베딩 벡터들을 인덱스에 추가합니다.
        print("FAISS 인덱스 구축 완료.")

    def _ensure_generator(self):
        """
        LLM과 토크나이저가 로드되었는지 확인하고, 로드되지 않았다면 로드합니다. (지연 로딩)
        이 함수는 실제로 답변 '생성'이 필요할 때만 호출되어 불필요한 메모리 사용을 방지합니다.
        """
        if self.generator is None:
            print(f"답변 생성 모델 '{self.llm_model_name}'을 로딩합니다 ({'GPU' if self.use_gpu else 'CPU'})...")
            
            # Hugging Face의 AutoModelForCausalLM을 사용하여 사전 학습된 LLM을 로드합니다.
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                # device_map="auto": 모델의 각 레이어를 여러 GPU 또는 CPU와 GPU에 걸쳐 최적으로 분산시켜 메모리 부족 문제를 해결합니다.
                device_map="auto" if self.use_gpu else None,
                # torch_dtype=torch.bfloat16: bfloat16 데이터 타입을 사용하여 모델을 로드합니다.
                # 이는 메모리 사용량을 절반으로 줄이고 계산 속도를 높여주지만, 약간의 정밀도 손실이 있을 수 있습니다. GPU에서만 사용 가능합니다.
                torch_dtype=torch.bfloat16 if self.use_gpu else None
            )
            # 모델에 맞는 토크나이저를 로드합니다. 토크나이저는 텍스트를 모델이 이해할 수 있는 숫자(토큰) 시퀀스로 변환합니다.
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            
            # text-generation 파이프라인을 생성하여 모델과 토크나이저를 하나로 묶어 쉽게 사용할 수 있게 합니다.
            self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            self.tokenizer = tokenizer # 토크나이저를 클래스 변수에도 저장하여 컨텍스트 길이 계산 등에 사용합니다.

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Tuple[int, float]]:
        """
        주어진 질문(query)에 대해 FAISS 인덱스에서 가장 유사한 top_k개의 문서를 검색합니다.
        - 반환값: (문서 인덱스, 유사도 점수)의 튜플 리스트
        """
        if self.index is None or self.embeddings is None: # 인덱스가 없으면 빈 리스트를 반환합니다.
            return []
            
        # 1. 사용자 질문을 임베딩 모델을 사용하여 벡터로 변환합니다.
        qv = self.embed_model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv).astype("float32")
        
        # 2. FAISS 인덱스에서 검색을 수행합니다.
        # D(Distances): 유사도 점수 배열. 여기서는 내적 값이므로 높을수록 유사합니다.
        # I(Indices): 검색된 문서의 인덱스 배열.
        D, I = self.index.search(qv, top_k)

        # 3. 결과를 (인덱스, 점수) 튜플의 리스트 형태로 가공하여 반환합니다.
        return list(zip(I[0], D[0]))

    def answer(self,
             query: str,
             top_k: int = TOP_K_RETRIEVE,
             score_threshold: float = DEFAULT_SCORE_THRESHOLD,
             max_return: int = 3,
             generate_answer: bool = False,
             gen_max_new_tokens: int = 2000,
             keyword_bonus: float = 0.2) -> Dict[str, Any]:
        """
        RAG 파이프라인의 전체 워크플로우를 실행하는 메인 메서드입니다.
        1. Retrieve: 질문과 유사한 문서를 검색합니다.
        2. Re-rank & Filter: 키워드 보너스를 적용하여 순위를 재조정하고, 점수 임계값으로 필터링합니다.
        3. (Optional) Generate: 필터링된 문서를 근거로 LLM을 통해 답변을 생성합니다.
        """
        # 1. [Retrieval] FAISS를 통해 1차적으로 유사한 문서를 검색합니다.
        hits_with_indices = self.retrieve(query, top_k=top_k)

        # 2. [Re-ranking] 키워드 보너스를 적용하여 순위를 재조정합니다.
        # 의미적 유사도(벡터 유사도)만으로는 부족할 수 있는 부분을 보완합니다.
        # 예를 들어, "연차 사용"이라는 질문에 "연차"라는 키워드가 제목에 포함된 규정이 더 중요할 수 있습니다.
        reranked_hits = []
        for idx, score in hits_with_indices:
            if idx < 0: continue # 유효하지 않은 인덱스는 건너뜁니다.

            bonus = 0.0
            keywords = self.chunk_keywords[idx] # 해당 문서의 제목에서 추출한 키워드
            for kw in keywords:
                if kw in query: # 키워드가 사용자 질문에 포함되어 있다면
                    bonus = keyword_bonus # 보너스 점수를 부여합니다.
                    break # 하나의 키워드만 일치해도 보너스를 주고 중단합니다.

            final_score = score + bonus # 최종 점수 = 의미 유사도 점수 + 키워드 보너스
            reranked_hits.append((idx, final_score))

        # 최종 점수가 높은 순으로 정렬합니다.
        reranked_hits.sort(key=lambda x: x[1], reverse=True)

        # 3. [Filtering] 초기 의미 유사도 점수가 임계값(score_threshold) 미만인 문서는 제외합니다.
        # 이는 키워드 보너스 때문에 관련성이 낮은 문서가 상위권에 오르는 것을 방지합니다.
        original_scores = {idx: score for idx, score in hits_with_indices} # 원래 점수를 쉽게 조회하기 위해 딕셔너리로 변환
        filtered_indices = [
            (idx, final_score) for idx, final_score in reranked_hits
            if original_scores.get(idx, 0) >= score_threshold
        ]

        # 4. 최종적으로 반환할 문서의 개수(max_return)만큼 선택합니다.
        final_results = filtered_indices[:max_return]

        # 5. 응답 객체를 구성합니다.
        response = {"query": query, "matches": []}
        final_chunks_for_llm = [] # LLM에게 전달할 근거 문서를 저장할 리스트

        for idx, final_score in final_results:
            chunk = self.chunks[idx]
            semantic_score = original_scores.get(idx, 0) # 키워드 보너스가 없는 순수 의미 점수

            # 미리보기용 텍스트 조각(snippet)을 생성합니다. (최대 800자)
            snippet = chunk["text"][:800].replace("\n", " ").strip()

            response["matches"].append({
                "title": chunk.get("title", ""),
                "semantic_score": round(semantic_score, 4), # 소수점 4자리까지 반올림
                "final_score": round(final_score, 4),
                "text_snippet": format_document_text(snippet),  # 가독성을 위해 포매팅 함수 적용
                "full_text": format_document_text(chunk["text"]) # 전체 텍스트도 포매팅
            })
            final_chunks_for_llm.append((chunk, final_score))

        # 6. [Generation] LLM 답변 생성 옵션이 활성화된 경우
        if generate_answer and self.llm_model_name:
            if not final_chunks_for_llm: # 근거로 삼을 문서가 없으면 생성하지 않음
                response["generated_answer"] = "관련 규정을 찾지 못해 답변을 생성할 수 없습니다."
            else:
                self._ensure_generator() # LLM이 로드되었는지 확인 및 로드

                # LLM에게 역할을 부여하고, 지시사항을 명확하게 전달하기 위한 프롬프트 템플릿입니다.
                # 이를 '프롬프트 엔지니어링'이라고 하며, LLM의 성능에 매우 중요한 영향을 미칩니다.
                prompt_template = (
                    "당신은 사내 규정 문서를 기반으로 답변하는 AI 어시스턴트입니다.\n"
                    "반드시 아래 제공된 '근거 문서'의 내용만을 사용하여 사용자 질의에 대해 답변을 요약하고 정리해야 합니다.\n"
                    "당신의 사전 지식이나 외부 정보를 절대로 사용하지 마십시오.\n"
                    "만약 근거 문서의 내용만으로 답변하기 어렵다면, '제공된 규정 내에서는 관련 내용을 찾을 수 없습니다.'라고만 답변하세요.\n\n"
                    "--- [사용자 질의] ---\n"
                    "{query}\n\n"
                    "--- [근거 문서] ---\n"
                    "{context_block}\n\n"
                    "--- [답변 요약] ---"
                )

                # LLM의 컨텍스트 길이(MODEL_MAX_CONTEXT) 제한을 초과하지 않도록 프롬프트에 포함할 문서의 양을 조절합니다.
                # 사용 가능한 토큰 수 = 최대 컨텍스트 길이 - (프롬프트 템플릿 길이 + 생성할 답변의 최대 길이 + 여유분)
                base_prompt_tokens = self.tokenizer.encode(
                    prompt_template.format(query=query, context_block="")
                )
                available_tokens = MODEL_MAX_CONTEXT - len(base_prompt_tokens) - gen_max_new_tokens - 20 # 20은 여유 토큰

                final_contexts = []
                current_tokens = 0
                for i, (c, s) in enumerate(final_chunks_for_llm, 1):
                    ctx_text = f"문서 {i}: {c['title']}\n{c['text']}"
                    ctx_tokens = self.tokenizer.encode(ctx_text)
                    if current_tokens + len(ctx_tokens) > available_tokens: # 사용 가능한 토큰 수를 초과하면 중단
                        print(f"경고: 모델의 컨텍스트 길이 제한에 도달하여 {i-1}번째 문서까지만 포함합니다.")
                        break
                    final_contexts.append(ctx_text)
                    current_tokens += len(ctx_tokens)

                ctx_block = "\n\n".join(final_contexts)
                prompt = prompt_template.format(query=query, context_block=ctx_block)

                # 구성된 프롬프트로 LLM 파이프라인을 호출하여 답변을 생성합니다.
                gen_out = self.generator(
                    prompt,
                    max_new_tokens=gen_max_new_tokens, # 생성할 최대 토큰 수
                    do_sample=True, # 샘플링 기반 생성을 활성화 (더 창의적인 답변 가능)
                    temperature=0.1, # 생성의 무작위성을 조절. 값이 낮을수록 결정론적이고 사실에 기반한 답변이 나옴. (사실 기반 요약이므로 0.1~0.3 권장)
                    repetition_penalty=1.1 # 같은 단어나 구문이 반복되는 것을 방지. (1.0 이상으로 설정)
                )

                raw = gen_out[0]["generated_text"]

                # LLM이 프롬프트 전체를 포함하여 답변을 생성하는 경우가 많으므로, 순수한 답변 부분만 추출합니다.
                answer_marker = "--- [답변 요약] ---"
                if answer_marker in raw:
                    raw = raw.split(answer_marker)[-1].strip()

                response["generated_answer"] = raw

        return response


# ====================== 스크립트 실행 부분 ======================
if __name__ == "__main__":
    """
    이 스크립트 파일이 직접 실행될 때만 이 코드 블록이 실행됩니다.
    다른 파일에서 이 스크립트를 'import'할 때는 실행되지 않습니다.
    """
    if not os.path.exists(CHUNKS_PATH):
        print(f"에러: 규정 파일('{CHUNKS_PATH}')을 찾을 수 없습니다.")
        print("스크립트와 같은 폴더에 파일이 있는지, 또는 경로 설정이 올바른지 확인해주세요.")
    else:
        # RAGAgent 클래스의 인스턴스를 생성합니다. 이 시점에서 모델 로딩 및 인덱스 구축이 이루어집니다.
        agent = RAGAgent(chunks_path=CHUNKS_PATH)

        # 사용자로부터 질문을 계속 입력받기 위한 무한 루프입니다.
        while True:
            q = input("\n질문을 입력하세요 (종료는 'exit'): ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break # 사용자가 'exit'를 입력하거나 그냥 엔터를 치면 루프를 종료합니다.
                
            # 에이전트의 answer 메서드를 호출하여 답변을 얻습니다.
            ans = agent.answer(q,
                               top_k=10,             # 초기 검색할 후보 수
                               score_threshold=0.45, # 최소 유사도 점수
                               max_return=3,         # 사용자에게 보여줄 최대 문서 수
                               generate_answer=True, # LLM 답변 생성 활성화
                               keyword_bonus=0.2,    # 키워드 일치 시 보너스 점수
                               gen_max_new_tokens=2000) # LLM이 생성할 최대 토큰 수

            # 결과를 출력합니다.
            print("\n\n" + "="*15 + " 규정 검색 결과 " + "="*15)
            if not ans["matches"]:
                print("일치하는 규정을 찾지 못했습니다.")
            else:
                for i, m in enumerate(ans["matches"], 1):
                    print(f"\n[{i}] 제목: {m['title']}  (최종점수: {m['final_score']}, 의미점수: {m['semantic_score']})")
                    # 버그 수정: print 문이 중복되어 있어 하나를 제거하고 올바르게 수정합니다.
                    # text_snippet은 이미 포매팅된 상태이므로 그대로 출력합니다.
                    print(f"  내용:\n{m['text_snippet']}")

            # 생성된 답변이 있다면 출력합니다.
            if "generated_answer" in ans:
                print("\n\n" + "="*15 + " 모델 생성 요약 답변 " + "="*15)
                # 버그 수정: print 문이 중복되어 있어 하나를 제거하고 올바르게 수정합니다.
                print(ans["generated_answer"])

            print("\n" + "="*42)