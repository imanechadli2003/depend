# rebuild_embeddings_50k.py
# ------------------------------------------------------------
# • Encode 50 000 passages (shard_docs_50k.txt)
# • Sauvegarde un memmap float32  (N, 384)  -> shard_embeds.npy
# ------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np, tqdm, pathlib

# --------- FICHIERS ET PARAMÈTRES ---------
DOC_FILE = "shard_docs_50k.txt"      # chaque ligne = passage complet
EMB_FILE = "shard_embeds.npy"        # sortie
BATCH    = 64                        # taille de batch SBERT

# --------- CHARGEMENT DU CORPUS ---------
with open(DOC_FILE, encoding="utf8") as f:
    passages = [line.strip().split('\t')[-1] for line in f]  # colonne passage_text

print(f"➜ {len(passages):,} passages à encoder")

# --------- MODÈLE SBERT ---------
model = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = model.get_sentence_embedding_dimension()          # 384 dims

print("➜ Dimension des embeddings :", EMB_DIM)

# --------- MEMMAP EN SORTIE (peu de RAM) ---------
emb = np.memmap(EMB_FILE,
                dtype="float32",
                mode="w+",
                shape=(len(passages), EMB_DIM))

# --------- ENCODAGE PAR LOTS ---------
for i in tqdm.trange(0, len(passages), BATCH, desc="Encoding"):
    emb[i:i+BATCH] = model.encode(
        passages[i:i+BATCH],
        batch_size=BATCH,
        normalize_embeddings=True,
        show_progress_bar=False
    )

print("✅  Embeddings terminés :", emb.shape, "➜", EMB_FILE)
