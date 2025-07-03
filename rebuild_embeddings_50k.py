from sentence_transformers import SentenceTransformer
import numpy as np, tqdm, os

DOC_FILE = "shard_docs_50k.txt"
EMB_FILE = "shard_embeds.npy"
BATCH    = 64

# Lire les passages
with open(DOC_FILE, encoding="utf8") as f:
    passages = [line.strip().split('\t')[-1] for line in f]

print("Passages :", len(passages))

# Modèle
model    = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM  = model.get_sentence_embedding_dimension()   # ➜ 384
print("Embedding dim :", EMB_DIM)

# Créer un memmap NEUF (shape = N, 384)
emb = np.memmap(EMB_FILE, dtype="float32",
                mode="w+", shape=(len(passages), EMB_DIM))

# Encodage
for i in tqdm.trange(0, len(passages), BATCH):
    emb[i:i+BATCH] = model.encode(
        passages[i:i+BATCH],
        batch_size=BATCH,
        normalize_embeddings=True)

print("✅ Terminé :", emb.shape)
