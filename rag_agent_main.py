#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG-based Internal Policy Agent
- 최대 3개 유사 조항 검색 및 선택적으로 모델 생성 답변
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import json
from typing import List, Dict, Any, Optional, Tuple
import re

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ====================== 사용자 설정 ======================
CHUNKS_PATH = "/home/alpaco/kimcy/Office_AI_Agent_System/report/standard_flattened.json"  # JSON 경로
EMBED_MODEL_NAME = "jhgan/ko-sbert-nli" 
LLM_MODEL_NAME = "skt/A.X-4.0-Light"
MODEL_MAX_CONTEXT = 16384
TOP_K_RETRIEVE = 10 
DEFAULT_SCORE_THRESHOLD = 0.45
# ========================================================

# ====[최종 수정] 전문가급 문서 포매팅 함수 구현====
def format_document_text(text: str) -> str:
    """
    규정 텍스트의 가독성을 높이기 위해 구조적으로 포매팅합니다.
    - ①, ② 등 '항'은 새로운 줄에서 시작합니다.
    - 1., 2. 등 '호'는 들여쓰기를 적용합니다.
    - 불필요한 공백과 줄바꿈을 정리합니다.
    """
    if not text:
        return ""

    # 1. 텍스트 초기 정규화
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 2. '항' (①, ② 등) 줄바꿈
    text = re.sub(r'\s*([②-⑩])\s*', r'\n\n\1 ', text)

    # 3. '호' (1., 2. 등) 들여쓰기
    text = re.sub(r'\s*(\d+\.)\s*', r'\n    \1 ', text)
    
    # 4. 일반 문장 끝 처리
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith('①') or re.match(r'^\d+\.', stripped)):
            line = line.replace('. ', '.\n')
        formatted_lines.append(line)
        
    return '\n'.join(formatted_lines)

class RAGAgent:
    def __init__(self,
                 chunks_path: str = CHUNKS_PATH,
                 embed_model_name: str = EMBED_MODEL_NAME,
                 llm_model_name: Optional[str] = LLM_MODEL_NAME,
                 use_gpu: bool = True):
        self.chunks_path = chunks_path
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(self.embed_model_name)

        self.chunks, self.chunk_keywords = self._load_chunks_and_keywords(self.chunks_path)
        self.embeddings = None
        self.index = None
        self._build_index()

        self.generator = None
        self.tokenizer = None
        
        
        print("RAG Agent ready.")

    def _load_chunks_and_keywords(self, path: str) -> Tuple[List[Dict[str, str]], List[List[str]]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunks file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # JSON 데이터가 딕셔너리 안에 "내용" 키로 래핑된 경우 처리 (하위 호환성)
        if isinstance(data, dict) and "내용" in data:
            data = data["내용"]

        if not isinstance(data, list) or not data:
            raise ValueError("JSON이 비어있거나 리스트 형식이 아닙니다.")

        chunks = []
        chunk_keywords = []

        # 하나의 반복문으로 chunks 와 chunk_keywords 모두 처리
        for i, item in enumerate(data):
            # 'text' 키는 필수, 없으면 에러 발생
            if "text" not in item:
                raise ValueError(f"{i+1}번째 항목에 필수 키인 'text'가 없습니다. (항목: {item})")

            # 'title' 키는 선택, 없으면 빈 문자열로 처리
            title = item.get("title", "")
            text = item["text"]
            
            # 텍스트에 실제 내용이 있을 경우에만 chunks와 keywords를 추가
            if text.strip():
                chunks.append({"title": title, "text": text})
                
                # --- 키워드 추출 로직 ---
                # "제n장", "제n조" 같은 패턴 제거
                clean_title = re.sub(r"제\d+장|제\d+조\s*", "", title).strip()
                # 특수 문자 공백으로 변경
                processed_title = clean_title.replace(" ․ ", " ").replace(" · ", " ")
                # 공백 기준으로 단어 분리하여 키워드 리스트 생성
                keywords = processed_title.split()
                # 각 키워드의 앞뒤 공백 제거 후, 빈 문자열이 아닌 것만 추가
                chunk_keywords.append([kw.strip() for kw in keywords if kw.strip()])

        if not chunks:
            raise ValueError("유효한 내용이 있는 청크가 없습니다.")
        
        print(f"Loaded {len(chunks)} chunks and extracted keywords.")
        return chunks, chunk_keywords

    def _build_index(self):
        if not self.chunks:
            self.index = None
            self.embeddings = None
            return
        texts = [c["text"] for c in self.chunks]
        print("Computing embeddings for chunks...")
        embs = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embs = np.asarray(embs).astype("float32")
        self.embeddings = embs
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        print("FAISS index built.")

    def _ensure_generator(self):
        # LLM 모델과 토크나이저를 실제로 사용할 때만 로딩합니다.
        if self.generator is None:
            print(f"Loading generation model '{self.llm_model_name}' on {'GPU' if self.use_gpu else 'CPU'} ...")
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map="auto" if self.use_gpu else None,
                torch_dtype=torch.bfloat16 if self.use_gpu else None
            )
            # 토크나이저는 여기서 모델과 함께 로딩하는 것이 가장 효율적입니다.
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            self.tokenizer = tokenizer # 클래스 변수에 할당

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Tuple[int, float]]:
        if self.index is None or self.embeddings is None:
            return []
        qv = self.embed_model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv).astype("float32")
        D, I = self.index.search(qv, top_k)
        
        return list(zip(I[0], D[0]))

    def answer(self,
             query: str,
             top_k: int = TOP_K_RETRIEVE,
             score_threshold: float = DEFAULT_SCORE_THRESHOLD,
             max_return: int = 3,
             generate_answer: bool = False,
             gen_max_new_tokens: int = 2000,
             keyword_bonus: float = 0.2) -> Dict[str, Any]:

        hits_with_indices = self.retrieve(query, top_k=top_k)

        reranked_hits = []
        for idx, score in hits_with_indices:
            if idx < 0: continue

            bonus = 1.0
            keywords = self.chunk_keywords[idx] # 해당 문서의 제목에서 추출한 키워드
            for kw in keywords:
                if kw in query: # 키워드가 사용자 질문에 포함되어 있다면
                    bonus += keyword_bonus # 보너스 점수를 부여합니다.
                    break # 하나의 키워드만 일치해도 보너스를 주고 중단합니다.

            final_score = score + bonus
            reranked_hits.append((idx, final_score))

        reranked_hits.sort(key=lambda x: x[1], reverse=True)

        original_scores = {idx: score for idx, score in hits_with_indices}
        filtered_indices = [
            (idx, final_score) for idx, final_score in reranked_hits 
            if original_scores.get(idx, 0) >= score_threshold
        ]
        
        final_results = filtered_indices[:max_return]

        response = {"query": query, "matches": []}
        final_chunks_for_llm = []
        
        for idx, final_score in final_results:
            chunk = self.chunks[idx]
            semantic_score = original_scores.get(idx, 0)
            
            snippet = chunk["text"][:800].replace("\n", " ").strip()
            
            response["matches"].append({
            "title": chunk.get("title",""),
            "semantic_score": round(semantic_score, 4),
            "final_score": round(final_score, 4),
            "text_snippet": format_document_text(snippet),  # 포매팅 적용
            "full_text": format_document_text(chunk["text"])  # 전체 텍스트도 포매팅
              })
            final_chunks_for_llm.append((chunk, final_score))

        if generate_answer and self.llm_model_name:
            if not final_chunks_for_llm:
                response["generated_answer"] = "관련 근거가 발견되지 않아 자동생성하지 않았습니다."
            else:
                self._ensure_generator()
                
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
                
                # BUG FIX: 이전 코드의 'q'는 이 메서드 범위에서 정의되지 않은 변수입니다. 'query'로 수정해야 합니다.
                base_prompt_tokens = self.tokenizer.encode(
                    prompt_template.format(query=query, context_block="")
                )
                
                available_tokens = MODEL_MAX_CONTEXT - len(base_prompt_tokens) - gen_max_new_tokens - 20 
                
                final_contexts = []
                current_tokens = 0
                for i, (c, s) in enumerate(final_chunks_for_llm, 1):
                    ctx_text = f"문서 {i}: {c['title']}\n{c['text']}"
                    ctx_tokens = self.tokenizer.encode(ctx_text)
                    if current_tokens + len(ctx_tokens) > available_tokens:
                        print(f"Warning: Context limit reached. Stopping at reference {i-1}.")
                        break
                    final_contexts.append(ctx_text)
                    current_tokens += len(ctx_tokens)

                ctx_block = "\n\n".join(final_contexts)
                prompt = prompt_template.format(query=query, context_block=ctx_block)

                gen_out = self.generator(
                    prompt,
                    max_new_tokens=gen_max_new_tokens,
                    do_sample=True,
                    temperature=0.1, # 사실 기반 요약이므로 0.1 ~ 0.3 사이의 낮은 값 권장
                    repetition_penalty=1.1
                )
                
                raw = gen_out[0]["generated_text"]
                
                # 모델이 프롬프트를 그대로 출력하는 경우를 안정적으로 제거하는 로직
                answer_marker = "--- [답변 요약] ---"
                if answer_marker in raw:
                    raw = raw.split(answer_marker)[-1].strip()
                
                response["generated_answer"] = raw

        return response

if __name__ == "__main__":
    if not os.path.exists(CHUNKS_PATH):
        print(f"Error: 사내 규정 파일('{CHUNKS_PATH}')을 찾을 수 없습니다.")
        print("스크립트와 같은 폴더에 파일이 있는지 확인해주세요.")
    else:
        agent = RAGAgent(chunks_path=CHUNKS_PATH)

        while True:
            q = input("\n질문을 입력하세요 (종료는 'exit'): ").strip()
            if not q or q.lower() in ("exit","quit"):
                break
            ans = agent.answer(q,
                             top_k=10,
                             score_threshold=0.45,
                             max_return=3,
                             generate_answer=True,
                             keyword_bonus=0.2,
                             gen_max_new_tokens=2000)
    
             # 결과를 출력합니다.
            print("\n\n" + "="*15 + " 규정 검색 결과 " + "="*15)
            if not ans["matches"]:
                print("일치하는 규정을 찾지 못했습니다.")
            else:
                for i, m in enumerate(ans["matches"], 1):
                    print(f"\n[{i}] 제목: {m['title']}  (최종점수: {m['final_score']}, 의미점수: {m['semantic_score']})")
                    # text_snippet은 이미 포매팅된 상태이므로 그대로 출력합니다.
                    print(f"  내용:\n{m['text_snippet']}")

            # 생성된 답변이 있다면 출력합니다.
            if "generated_answer" in ans:
                print("\n\n" + "="*15 + " 모델 생성 요약 답변 " + "="*15)
                print(ans["generated_answer"])

            print("\n" + "="*42)