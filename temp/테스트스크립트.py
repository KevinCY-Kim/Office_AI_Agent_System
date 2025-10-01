ds# JSON 청크를 불러와서 인덱스에 바로 반영
# Dense (SentenceTransformer + Faiss)와 BM25를 동시에 테스트
# 테스트 쿼리 3개로 Hybrid 검색 결과 확인 가능
# 출력에서 청크 앞부분만 보여서 눈으로 확인 가능

import json
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# ===============================
# 1. JSON 청크 로드
# ===============================
rag_chunks_path = "/home/alpaco/kimcy/Office_AI_Agent_System/report/rag_chunks.json"
with open(rag_chunks_path, "r", encoding="utf-8") as f:
    rag_chunks_list = json.load(f)

chunks = [{"title": f"chunk_{i+1}", "text": txt} for i, txt in enumerate(rag_chunks_list)]
print(f"✅ 총 {len(chunks)}개의 청크 로드 완료\n")

print("=== 첫 3개 청크 미리보기 ===")
for c in chunks[:3]:
    print(f"[{c['title']}] {c['text'][:300]}...\n")

# ===============================
# 2. Dense 인덱스 생성
# ===============================
class TextIndexer:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)
        self.idx = None
        self.chunks = []

    def build(self):
        texts = [c["text"] for c in self.chunks]
        if not texts:
            self.idx = None
            return
        embs = self.model.encode(texts, normalize_embeddings=True)
        dim = embs.shape[1]
        self.idx = faiss.IndexFlatIP(dim)
        self.idx.add(embs.astype("float32"))

    def search_dense(self, query: str, topk=4):
        if self.idx is None or not self.chunks:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        D, I = self.idx.search(qv, topk)
        items = []
        for i in I[0]:
            if i < 0:
                continue
            items.append(self.chunks[i])
        return items

indexer = TextIndexer()
indexer.chunks = chunks
indexer.build()
print("✅ Dense 인덱스 생성 완료\n")

# ===============================
# 3. Hybrid Retriever
# ===============================
class HybridRetriever:
    def __init__(self, dense_indexer):
        self.dense = dense_indexer
        self.corpus = [c["text"] for c in self.dense.chunks]
        self.bm25 = BM25Okapi([t.split() for t in self.corpus]) if self.corpus else None

    def search(self, query, k_dense=4, k_bm25=4):
        dense_hits = self.dense.search_dense(query, topk=k_dense)
        bm_hits = []
        if self.bm25 is not None:
            bm_ids = self.bm25.get_top_n(query.split(), list(range(len(self.corpus))), n=k_bm25)
            bm_hits = [self.dense.chunks[i] for i in bm_ids]
        seen = set()
        results = []
        for it in dense_hits + bm_hits:
            key = it["text"][:80]
            if key in seen: 
                continue
            seen.add(key)
            results.append(it)
        return results[:max(k_dense, k_bm25)]

retriever = HybridRetriever(indexer)
print("✅ Hybrid RAG 검색기 준비 완료\n")

# ===============================
# 4. 검색 테스트
# ===============================
test_queries = [
    "사업 개요",
    "기술 해결 방안",
    "경영 전략과 기대 효과"
]

for query in test_queries:
    hits = retriever.search(query, k_dense=2, k_bm25=2)
    print(f"\n=== Hybrid RAG 검색: '{query}' ===")
    for i, r in enumerate(hits):
        print(f"[{i+1}] {r['title']}: {r['text'][:200]}...\n")
