from typing import List, Dict

def format_prompt(instruction: str, query: str, contexts: List[Dict]) -> str:
    ctx = "\n\n".join([f"[근거]\n{c['text']}" for c in contexts])
    return f"""아래 근거 텍스트 안에서만 답변해.
- 출처 밖 가정 금지
- 수치/날짜/명칭은 원문 그대로
- 최대 2~3문장으로 간결하게

[요청]
{instruction}

[검색 질의]
{query}

{ctx}
"""

def call_local_llm(prompt: str) -> str:
    # TODO: skt/A.X-4.0-Light 연동 (vLLM/llama.cpp 등)
    # 예: requests.post("http://localhost:8000/generate", json={...})
    return "LLM 응답(스텁): 로컬 LLM 엔드포인트 연결 시 실제 결과가 출력됩니다."

def generate(instruction: str, query: str, contexts: List[Dict]) -> str:
    prompt = format_prompt(instruction, query, contexts)
    return call_local_llm(prompt)