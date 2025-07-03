import faiss, numpy as np
from sentence_transformers import SentenceTransformer

idx   = faiss.read_index("shard_hnsw.faiss")
docs  = [l.strip().split('\t')[-1] for l in open("shard_docs_50k.txt")]
model = SentenceTransformer("all-MiniLM-L6-v2")

q  = "Who discovered penicillin?"
qv = model.encode(q, normalize_embeddings=True).reshape(1, -1)
_, I = idx.search(qv, 3)
print("🔎 Meilleur passage :\n", docs[I[0][0]][:200])
