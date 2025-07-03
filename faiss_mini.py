# build_faiss_mini.py
# ----------------------------------------------------------
# • Charge shard_embeds.npy   (50 000 × 384 float32)
# • Normalise L2
# • Construit un index FAISS HNSW-32
# • Sauvegarde shard_hnsw.faiss
# ----------------------------------------------------------

import numpy as np
import faiss
import os

EMB_FILE   = "shard_embeds.npy"   # sortie du script SBERT
FAISS_FILE = "shard_hnsw.faiss"
HNSW_M     = 32                   # nb voisins / nœud
print("🟡  Lecture des embeddings :", EMB_FILE)

# 1) charger le memmap
try:
    vecs = np.load(EMB_FILE, mmap_mode="r")
except ValueError:
    # fichier enregistré avec pickle -> autoriser explicitement
    vecs = np.load(EMB_FILE, mmap_mode="r", allow_pickle=True)

print("   Shape :", vecs.shape, "| dtype :", vecs.dtype)

# 2) normalisation L2
vecs = vecs.astype("float32")          # FAISS exige float32
faiss.normalize_L2(vecs)

# 3) création de l'index HNSW
d = vecs.shape[1]                      # 384 dims
index = faiss.IndexHNSWFlat(d, HNSW_M)

print("🟡  Ajout des vecteurs dans l'index…")
index.add(vecs)                        # ∼ 30 s pour 50 000 vecteurs

# 4) sauvegarde
faiss.write_index(index, FAISS_FILE)
print(f"✅  Index FAISS écrit : {FAISS_FILE} — {index.ntotal:,} vecteurs")
