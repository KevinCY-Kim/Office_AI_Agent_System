from PyPDF2 import PdfReader, PdfWriter

reader = PdfReader("/home/alpaco/kimcy/Office_AI_Agent_System/report/VUNO_business_report_2025.pdf")
total_pages = len(reader.pages)
split_size = 100  # 100페이지씩

for i in range(0, total_pages, split_size):
    writer = PdfWriter()
    for j in range(i, min(i + split_size, total_pages)):
        writer.add_page(reader.pages[j])
    temp_pdf = f"report_part_{i//split_size + 1}.pdf"
    with open(temp_pdf, "wb") as f:
        writer.write(f)
    print(f"생성: {temp_pdf}")