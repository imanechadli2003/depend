import numpy as np
from sentence_transformers import SentenceTransformer

with open("shard_docs.txt", "r", encoding="utf8") as f:
    docs = [line.strip() for line in f]

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeds = encoder.encode(docs, batch_size=64, show_progress_bar=True)
np.save("shard_embeds.npy", embeds)
print("Saved new shard_embeds.npy with", embeds.shape[0], "docs and", embeds.shape[1], "dims.")
