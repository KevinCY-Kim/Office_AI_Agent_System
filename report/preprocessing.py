import json
import re

def clean_text(text: str) -> str:
    # 숫자 리스트(1., 2., …) 앞에 줄바꿈 추가
    text = re.sub(r'(\d+\.)', r'\n\1', text)
    # 조항 기호(①, ②, …) 앞에 줄바꿈 추가
    text = re.sub(r'([①②③④⑤⑥⑦⑧⑨])', r'\n\1', text)
    # 불필요한 여러 개 줄바꿈은 하나로 줄이기
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# 파일 경로
file_path = "/home/alpaco/kimcy/Office_AI_Agent_System/report/standard_flattened.json"

# JSON 불러오기
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 데이터 형태가 리스트일 경우
if isinstance(data, list):
    for item in data:
        if "text" in item and isinstance(item["text"], str):
            item["text"] = clean_text(item["text"])

# 데이터 형태가 dict일 경우
elif isinstance(data, dict):
    if "text" in data and isinstance(data["text"], str):
        data["text"] = clean_text(data["text"])

# 변환된 JSON 저장
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 파일 정제 완료:", file_path)