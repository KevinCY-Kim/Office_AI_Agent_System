import pdfplumber
import pandas as pd

TABLE_KEYWORDS = ["주주명", "지분율", "종속기업", "결산월", "소유지분율", "재무상태표", "포괄손익계산서"]

def extract_priority_tables(pdf_path: str) -> dict:
    tables = {"shareholders": [], "subsidiaries": [], "finance": []}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                tb = page.extract_tables()
            except Exception:
                tb = []
            for t in tb or []:
                df = pd.DataFrame(t)
                head = " ".join(df.iloc[0].astype(str).tolist()).lower() if len(df) > 0 else ""
                if any(k in head for k in ["주주명", "지분율"]):
                    tables["shareholders"].append(df)
                elif any(k in head for k in ["종속기업", "소유지분율", "결산월"]):
                    tables["subsidiaries"].append(df)
                elif any(k in head for k in ["재무상태표", "포괄손익", "현금흐름표"]):
                    tables["finance"].append(df)
    return tables

def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0:
        return df
    df.columns = [str(c).strip() for c in df.iloc[0]]
    df = df[1:].reset_index(drop=True)
    return df