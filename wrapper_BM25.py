from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

es    = Elasticsearch("http://localhost:9200")          # index BM25 'wiki'
idx   = faiss.read_index("shard_hnsw.faiss")            # nouvel index dense
model = SentenceTransformer("all-MiniLM-L6-v2")

def hybrid_topk(question, k_bm25=1000, k_dense=256):
    # 1. rappel lexical
    hits = es.search(index="wiki", size=k_bm25,
                     query={"match": {"passage_text": question}})
    passages = [h["_source"]["passage_text"] for h in hits["hits"]["hits"]]

    # 2. encode ces passages + recherche dense sur mini-index
    vecs = model.encode(passages, normalize_embeddings=True)
    sub  = faiss.IndexFlatIP(384); sub.add(vecs.astype("float32"))
    qv   = model.encode(question, normalize_embeddings=True).reshape(1,-1)
    _, I = sub.search(qv, k_dense)

    return [passages[i] for i in I[0]], (vecs[I[0]] @ qv.T).flatten()
