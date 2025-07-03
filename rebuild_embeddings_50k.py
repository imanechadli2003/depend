# rebuild_embeddings_50k.py  — version 100 % robuste
from sentence_transformers import SentenceTransformer
import numpy as np, tqdm, os

DOC_FILE = "shard_docs_50k.txt"      # 50 000 lignes
EMB_FILE = "shard_embeds.npy"        # sortie finale
BATCH    = 64

# 1) charger les passages
with open(DOC_FILE, encoding="utf8") as f:
    passages = [l.strip().split('\t')[-1] for l in f]

print("Passages à encoder :", len(passages))           # doit afficher 50000

# 2) modèle SBERT
model    = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM  = model.get_sentence_embedding_dimension()    # 384

# 3) repartir de zéro : effacer l'ancien fichier s'il existe
if os.path.exists(EMB_FILE):
    os.remove(EMB_FILE)

# 4) créer le memmap
emb = np.memmap(EMB_FILE, dtype="float32",
                mode="w+", shape=(len(passages), EMB_DIM))

# 5) encodage par lots (sans déborder)
for i in tqdm.trange(0, len(passages), BATCH, desc="Encoding"):
    end = min(i + BATCH, len(passages))                # coupe le dernier lot
    emb[i:end] = model.encode(passages[i:end],
                              batch_size=BATCH,
                              normalize_embeddings=True)

print("✅ Terminé :", emb.shape, "embeddings écrits dans", EMB_FILE)
