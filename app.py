import argparse, os
from rag.ingest import load_pdf_text, clean_text
from rag.anchors import segment_by_anchors
from rag.indexer import TextIndexer
from rag.retriever import HybridRetriever
from rag.generator import generate
from rag.cache import cache_key, get_cache, set_cache

def main(pdf_path: str, query: str, instruction: str | None, use_cache=True):
    key = cache_key(pdf_path, query)
    cache_path = f"./cache/{key}.json"
    if use_cache:
        cached = get_cache(cache_path)
        if cached:
            print("[CACHE HIT]")
            print(cached["answer"])
            return

    print("[INGEST] PDF 로드 및 정제")
    raw = load_pdf_text(pdf_path)
    text = clean_text(raw)

    print("[ANCHOR] 섹션 분할")
    segments = segment_by_anchors(text)

    print("[INDEX] 임베딩 인덱스 생성")
    indexer = TextIndexer()
    chunks = indexer.chunk(segments, max_chars=1200)
    indexer.build()

    print("[RETRIEVE] 온디맨드 하이브리드 검색")
    retriever = HybridRetriever(indexer, chunks)
    contexts = retriever.search(query, k_dense=4, k_bm25=4)

    print("[GENERATE] LLM 생성 호출")
    ans = generate(instruction or "질의에 맞춰 간결히 요약하라.", query, contexts)
    out = {"answer": ans, "used_chunks": [c["title"] for c in contexts]}

    if use_cache:
        set_cache(cache_path, out)

    print(ans)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    main(args.pdf, args.query, args.instruction, use_cache=not args.no_cache)