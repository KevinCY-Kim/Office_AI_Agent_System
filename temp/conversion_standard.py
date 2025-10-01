import json
import os

INPUT_PATH = "/home/alpaco/kimcy/Office_AI_Agent_System/report/standard.json"
OUTPUT_PATH = "/home/alpaco/kimcy/Office_AI_Agent_System/report/standard_flattened.json"

def flatten_standard_json(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for chapter in data.get("내용", []):
        chapter_title = chapter.get("장", "")
        for clause in chapter.get("조항", []):
            clause_number = clause.get("조항번호", "")
            clause_title = clause.get("제목", "")
            text = clause.get("내용", "")
            # title 예: "제1장 총칙 - 제1조 목적"
            full_title = f"{chapter_title} - {clause_number} {clause_title}"
            chunks.append({"title": full_title, "text": text})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Flattened {len(chunks)} chunks and saved to {output_path}")

if __name__ == "__main__":
    flatten_standard_json(INPUT_PATH, OUTPUT_PATH)