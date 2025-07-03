# fichier : rebuild_embeddings_50k.py
from sentence_transformers import SentenceTransformer
import numpy as np, tqdm, pathlib

DOC_FILE = "shard_docs_50k.txt"
EMB_FILE = "shard_embeds.npy"
BATCH    = 64

texts = [l.strip().split('\t')[-1] for l in open(DOC_FILE)]   # prend le passage_text
model = SentenceTransformer("all-MiniLM-L6-v2")

emb = np.memmap(EMB_FILE, dtype="float32", mode="w+", shape=(len(texts), 768))

for i in tqdm.trange(0, len(texts), BATCH):
    emb[i:i+BATCH] = model.encode(
        texts[i:i+BATCH],
        batch_size=BATCH,
        normalize_embeddings=True)

print("✅ embeddings sauvegardés :", emb.shape, EMB_FILE)