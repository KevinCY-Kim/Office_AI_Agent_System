from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict

class TextIndexer:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)
        self.idx = None
        self.chunks = []

    def chunk(self, segments: List[Dict], max_chars=1200):
        res = []
        for seg in segments:
            buf = []
            acc = 0
            for line in seg["text"].split("\n"):
                if not line.strip():
                    continue
                if acc + len(line) > max_chars and buf:
                    res.append({"title": seg["title"], "text": " ".join(buf)})
                    buf, acc = [], 0
                buf.append(line.strip())
                acc += len(line)
            if buf:
                res.append({"title": seg["title"], "text": " ".join(buf)})
        self.chunks = res
        return res

    def build(self):
        texts = [c["text"] for c in self.chunks]
        embs = self.model.encode(texts, normalize_embeddings=True)
        dim = embs.shape[1]
        self.idx = faiss.IndexFlatIP(dim)
        self.idx.add(embs.astype("float32"))

    def search(self, query: str, topk=5):
        qv = self.model.encode([query], normalize_embeddings=True)
        D, I = self.idx.search(qv, topk)
        items = []
        for i in I[0]:
            if i < 0:
                continue
            items.append(self.chunks[i])
        return items