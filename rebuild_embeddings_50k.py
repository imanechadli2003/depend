# encode_full_robust.py
from sentence_transformers import SentenceTransformer
import numpy as np, tqdm, os

DOC_FILE = "shard_docs_50k.txt"
EMB_FILE = "shard_embeds.npy"
BATCH    = 32               # ↓ si GPU/CPU limite
model    = SentenceTransformer("all-MiniLM-L6-v2")
DIM      = model.get_sentence_embedding_dimension()

passages = [l.strip().split('\t')[-1] for l in open(DOC_FILE)]
N = len(passages)

# (ré)crée le memmap
emb = np.memmap(EMB_FILE, dtype='float32', mode='w+', shape=(N, DIM))

start = 0
for i in range(0, N, BATCH):
    end = min(i+BATCH, N)
    emb[i:end] = model.encode(
        passages[i:end],
        batch_size=BATCH,
        normalize_embeddings=True
    )
    if (i // BATCH) % 50 == 0:     # flush toutes les 50 itérations
        emb.flush()

emb.flush(); del emb
print("✅ embeddings final :", np.load(EMB_FILE, mmap_mode='r').shape)
