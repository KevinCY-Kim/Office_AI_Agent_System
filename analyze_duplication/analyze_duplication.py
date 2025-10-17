import os
import docx
import re  # kss 대신 정규표현식 라이브러리 import
from sentence_transformers import SentenceTransformer, util
import torch

# --- 설정 (이 부분을 수정하여 사용하세요) ---
MODEL_NAME = 'jhgan/ko-sroberta-multitask'
SIMILARITY_THRESHOLD = 0.75

# --- 코드 본문 ---

def read_text_from_docx(file_path):
    """ .docx 파일에서 모든 텍스트를 추출합니다. """
    try:
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(full_text)
    except Exception as e:
        print(f"오류: 파일을 읽을 수 없습니다. 경로를 확인하세요. ({e})")
        return None

def analyze_document_duplication(file_path):
    """
    주어진 docx 파일 내 문장들의 의미론적 중복도를 분석합니다.
    """
    print(f"'{file_path}' 문서 분석을 시작합니다...")
    print("-" * 50)

    # 1. 문서 읽기 및 문장 분리
    document_text = read_text_from_docx(file_path)
    if not document_text:
        return

    # ===================================================================
    # [최종 수정] kss 라이브러리를 re(정규표현식)로 완전 대체
    # ===================================================================
    print("파이썬 내장 기능으로 문장 분리를 실행합니다...")
    # 여러 줄의 텍스트를 한 줄로 합치고, 마침표(.), 물음표(?), 느낌표(!)를 기준으로 문장 분리
    processed_text = document_text.replace('\n', ' ').replace('\r', ' ')
    sentences = re.split(r'(?<=[.?!])\s+', processed_text)
    # ===================================================================
    
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) < 2:
        print("분석할 문장이 2개 미만입니다. 분석을 종료합니다.")
        return

    print(f"1. 총 {len(sentences)}개의 유효 문장을 추출했습니다.")

    # 2. SBERT 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"2. '{MODEL_NAME}' 모델을 로드했습니다. (Device: {device})")

    # 3. 모든 문장을 벡터로 변환 (임베딩)
    print("3. 문장을 의미 벡터로 변환하는 중입니다... (시간이 걸릴 수 있습니다)")
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

    # 4. 모든 문장 쌍 간의 코사인 유사도 계산
    duplicate_pairs_count = 0
    total_comparisons = 0
    duplicate_examples = []

    print("4. 문장 쌍 간의 유사도를 계산하고 중복을 검출합니다...")
    cosine_scores = util.cos_sim(embeddings, embeddings)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            total_comparisons += 1
            score = cosine_scores[i][j].item()
            
            if score > SIMILARITY_THRESHOLD:
                duplicate_pairs_count += 1
                if len(duplicate_examples) < 5:
                    duplicate_examples.append((score, sentences[i], sentences[j]))

    # 5. 최종 결과 계산 및 출력
    duplication_rate = (duplicate_pairs_count / total_comparisons) * 100 if total_comparisons > 0 else 0

    print("\n" + "=" * 50)
    print("🎉 분석 완료! 최종 결과 🎉")
    print("=" * 50)
    print(f"  - 총 비교 문장 쌍 수: {total_comparisons:,} 개")
    print(f"  - 유사도 임계값: {SIMILARITY_THRESHOLD}")
    print(f"  - 발견된 의미 중복 쌍 수: {duplicate_pairs_count:,} 개")
    print(f"  - 📈 최종 문서 내 중복률: {duplication_rate:.2f}%")
    print("-" * 50)

    if duplicate_examples:
        print("\n[ 참고: 발견된 중복 문장 예시 (상위 5개) ]\n")
        for score, s1, s2 in sorted(duplicate_examples, key=lambda x: x[0], reverse=True):
            print(f"▶ 유사도: {score:.2f}")
            print(f"  - 문장 1: {s1}")
            print(f"  - 문장 2: {s2}\n")


if __name__ == '__main__':
    target_file = "/home/alpaco/kimcy/Office_AI_Agent_System/develop_report/CDEv2_계획서0930.docx" 
    analyze_document_duplication(target_file)