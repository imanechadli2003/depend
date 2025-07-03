from sentence_transformers import SentenceTransformer
import numpy as np, tqdm, os

DOC_FILE = "shard_docs_50k.txt"
EMB_FILE = "shard_embeds.npy"
BATCH    = 64

passages = [l.strip().split('\t')[-1] for l in open(DOC_FILE)]
model    = SentenceTransformer("all-MiniLM-L6-v2")
DIM      = model.get_sentence_embedding_dimension()   # 384

emb = np.memmap(EMB_FILE, dtype="float32",
                mode="w+", shape=(len(passages), DIM))

for i in tqdm.trange(0, len(passages), BATCH, desc="Encoding"):
    end = min(i+BATCH, len(passages))
    emb[i:end] = model.encode(
        passages[i:end],
        batch_size=BATCH,
        normalize_embeddings=True)

emb.flush()                       # <<< vide le cache sur disque
del emb                           # libère le memmap

print("✅ Embeddings écrits :", EMB_FILE)
