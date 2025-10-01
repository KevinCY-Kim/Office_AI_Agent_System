import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

# ===========================
# 1. API 설정
# ===========================
API_KEY = "YOUR_API_KEY"
URL = "https://api.upstage.ai/v1/document-parse"
PDF_PATH = "report.pdf"
CHUNK_SIZE = 500  # 단어 기준

# ===========================
# 2. PDF 페이지 단위 분리
# ===========================
reader = PdfReader(PDF_PATH)
total_pages = len(reader.pages)
print(f"총 {total_pages} 페이지")

# ===========================
# 3. 페이지별 API 호출 + 청킹
# ===========================
all_chunks = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if not text or text.strip() == "":
        print(f"페이지 {i+1}: 텍스트 없음, 스킵")
        continue

    # 페이지 텍스트를 임시 파일로 저장
    temp_pdf_path = f"temp_page_{i+1}.pdf"
    # PyPDF2를 사용하여 단일 페이지 PDF 생성
    from PyPDF2 import PdfWriter
    writer = PdfWriter()
    writer.add_page(page)
    with open(temp_pdf_path, "wb") as f:
        writer.write(f)

    # API 호출
    with open(temp_pdf_path, "rb") as f:
        files = {"file": f}
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post(URL, headers=headers, files=files)

    if response.status_code != 200:
        print(f"페이지 {i+1} API 실패: {response.status_code}")
        continue

    html_result = response.json().get("html", "")
    soup = BeautifulSoup(html_result, "html.parser")
    page_lines = [line.strip() for line in soup.get_text(separator="\n").split("\n") if line.strip()]

    # 페이지 단위 청킹
    current_chunk = []
    for line in page_lines:
        words = line.split()
        while words:
            remaining_space = CHUNK_SIZE - len(current_chunk)
            current_chunk.extend(words[:remaining_space])
            words = words[remaining_space:]
            if len(current_chunk) >= CHUNK_SIZE:
                all_chunks.append(" ".join(current_chunk))
                current_chunk = []
    if current_chunk:
        all_chunks.append(" ".join(current_chunk))

    print(f"페이지 {i+1} 완료, 총 청크 수: {len(all_chunks)}")

print(f"총 생성 청크: {len(all_chunks)}")
