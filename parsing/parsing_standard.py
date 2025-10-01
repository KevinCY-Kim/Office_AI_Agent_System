# pip install requests

import requests
import json

api_key = "up_HOZmYLrQeFZpUMRkapN9nKa47G33P"   # ex: up_xxxYYYzzzAAAbbbCCC
filename = "/home/alpaco/kimcy/Office_AI_Agent_System/2023_standard_convertion1.docx"  # ex: ./report.pdf

# ===========================
# 1. API 호출
# ===========================
url = "https://api.upstage.ai/v1/document-digitization"
headers = {"Authorization": f"Bearer {api_key}"}
files = {"document": open(filename, "rb")}
data = {
    "ocr": "force", 
    "base64_encoding": "['table']", 
    "model": "document-parse"
}

response = requests.post(url, headers=headers, files=files, data=data)

# ===========================
# 2. API 응답 확인
# ===========================
if response.status_code == 200:
    response_data = response.json()
    print("파싱 완료, JSON 저장 준비")
else:
    raise Exception(f"API 호출 실패: {response.status_code} {response.text}")

# ===========================
# 3. JSON 저장
# ===========================
json_filename = "parsed_standard.json"
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(response_data, f, ensure_ascii=False, indent=2)

print(f"파싱 결과를 '{json_filename}'로 저장 완료")