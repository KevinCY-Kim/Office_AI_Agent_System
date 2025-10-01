import os, orjson, hashlib

def cache_key(pdf_path: str, query: str) -> str:
    base = f"{os.path.basename(pdf_path)}::{query}"
    return hashlib.md5(base.encode()).hexdigest()

def get_cache(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def set_cache(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))