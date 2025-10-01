import os, json, random, math
import numpy as np
import torch
from tqdm.auto import tqdm
import re # 정규표현식 사용을 위해 추가
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
)
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn

from datasets import Dataset
from torch.utils.data import DataLoader
from typing import List

# ==============================================================================
# 1. 환경 설정 및 경로 변수
# ==============================================================================
GUIDELINE_FILE = '/home/alpaco/kimcy/Office_AI_Agent_System/config/rnd_guidelines/rnd_guidline.json'
RAG_JSON_FILES = ["/home/alpaco/kimcy/Office_AI_Agent_System/chunking/rag_chunks500_50.json"] 

# 사용 모델 명
E5_NAME = "intfloat/multilingual-e5-base" # e5 모델 (검색 관련 로직은 제거됨)
GEN_NAME = "skt/A.X-4.0-Light"              # 생성 LLM

# 하드웨어 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_ID = 0 if torch.cuda.is_available() else -1    

# e5 모델 관련 파라미터 (검색이 비활성화되어 사용되지 않음)
DOCUMENT_PREFIX = "passage: " 
QUERY_PREFIX    = "query: "   
MAXLEN_DOC   = 512            
MAXLEN_QUERY = 512            
BATCH_SIZE   = 32             

GEN_MODE = "pipeline_batch" 
GEN_MAX_NEW_TOKENS = 5000
GEN_DO_SAMPLE = False           
GEN_TEMPERATURE = None          


# ==============================================================================
# 2. 유틸리티 함수
# ==============================================================================
def l2_normalize(t: torch.Tensor, dim=1):
    """L2 정규화 함수 (현재 코드에서는 사용되지 않음)"""
    return torch.nn.functional.normalize(t, p=2, dim=dim)

def first_n_lines(text: str, n_chars=350):
    """텍스트를 띄어쓰기로 합치고 처음 n_chars만큼 반환 (근거 텍스트 간결화)"""
    t = " ".join(str(text).split())
    return t[:n_chars]

def clean_generated_text(text: str) -> str:
    """
    생성된 텍스트에서 프롬프트 부분('#=========== 출력' 이전)과 지정된 Markdown 포맷을 제거합니다.
    """
    # 1. 프롬프트 헤더 이후의 텍스트만 추출
    output_marker = "#=========== 출력"
    if output_marker in text:
        text = text.split(output_marker, 1)[-1].strip()

    # 2. 불용어 및 Markdown 포맷 제거
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = text.replace('**', '')
    

    return text.strip()

# ==============================================================================
# 3. 데이터 로드 (FAISS 인덱스 구축 로직 제거)
# ==============================================================================
# 1) 가이드라인 및 RAG 데이터 로드
print("[DEBUG] open target =", GUIDELINE_FILE, type(GUIDELINE_FILE))

with open(GUIDELINE_FILE, "r", encoding="utf-8") as f:
    # R&D 가이드라인 데이터 로드 (프롬프트 구성에 사용)
    guidelines = json.load(f)

combined_text = ""
for jf in RAG_JSON_FILES:
    with open(jf, "r", encoding="utf-8") as f:
        parsed = json.load(f)
        # RAG 데이터 JSON 파일을 순회하며 텍스트를 결합 (컨텍스트/근거로 활용)
        if isinstance(parsed, dict) and "text" in parsed:
            combined_text += parsed["text"] + "\n\n"
        else:
            combined_text += str(parsed) + "\n\n"
print("[INFO] RAG chunks loaded. (Only combined_text will be used as context.)")
