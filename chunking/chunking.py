import json
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===========================
# 1. JSON 파일 목록
# ===========================
json_files = ["/home/alpaco/kimcy/Office_AI_Agent_System/report/parsed_report1.json", 
              "/home/alpaco/kimcy/Office_AI_Agent_System/report/parsed_report2.json", 
              "/home/alpaco/kimcy/Office_AI_Agent_System/report/parsed_report3.json"]

# ===========================
# 2. 청킹 설정
# ===========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 청크 크기
    chunk_overlap=50,        # 청크 겹침
    separators=["\n\n", "\n", ".", "!", "?", " "]  # 큰 단위 → 작은 단위
)

all_chunks = []

# ===========================
# 3. 각 JSON 파일 처리
# ===========================
for file in json_files:
    print(f"Processing {file} ...")
    with open(file, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    # HTML 추출
    html_content = parsed_data.get("content", {}).get("html", "")
    if not html_content:
        print(f"{file}에 HTML 없음, 스킵")
        continue

    soup = BeautifulSoup(html_content, "html.parser")

    # ===========================
    # 4. 문단/제목 텍스트 추출
    # ===========================
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3","h4","h5","h6"])]

    # ===========================
    # 5. 표(table) 텍스트 추출
    # ===========================
    tables_text = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            row_text = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            if row_text:
                tables_text.append(" | ".join(row_text))

    # ===========================
    # 6. 모든 텍스트 합치기
    # ===========================
    all_text_lines = headings + paragraphs + tables_text
    merged_text = "\n".join(all_text_lines)

    # ===========================
    # 7. 청킹 (RecursiveCharacterTextSplitter)
    # ===========================
    chunks = text_splitter.split_text(merged_text)
    all_chunks.extend(chunks)

print(f"총 {len(all_chunks)}개의 청크 생성 완료")

# ===========================
# 8. 청크 저장
# ===========================
with open("rag_chunks500_50.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("RAG용 청크를 'rag_chunks500_50.json'로 저장 완료")
