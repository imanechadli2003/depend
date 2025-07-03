import faiss, numpy as np

vecs = np.load("shard_embeds.npy", mmap_mode="r").astype("float32")
faiss.normalize_L2(vecs)                      # sécurité
index = faiss.IndexHNSWFlat(384, 32)          # parfait < 100 k vecteurs
index.add(vecs)
faiss.write_index(index, "shard_hnsw.faiss")
print("✅  Index FAISS prêt :", index.ntotal, "vecteurs")
