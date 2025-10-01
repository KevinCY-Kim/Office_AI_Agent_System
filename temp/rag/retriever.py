from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, dense_indexer, chunks):
        self.dense = dense_indexer
        self.chunks = chunks
        self.corpus = [c["text"] for c in chunks]
        self.bm25 = BM25Okapi([t.split() for t in self.corpus])

    def search(self, query, k_dense=4, k_bm25=4):
        dense_hits = self.dense.search(query, topk=k_dense)
        bm_ids = self.bm25.get_top_n(query.split(), list(range(len(self.corpus))), n=k_bm25)
        bm_hits = [self.chunks[i] for i in bm_ids]
        seen = set()
        results = []
        for it in dense_hits + bm_hits:
            key = it["text"][:60]
            if key in seen:
                continue
            seen.add(key)
            results.append(it)
        return results[:max(k_dense, k_bm25)]