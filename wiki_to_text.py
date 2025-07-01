input_tsv = "./data/dpr/psgs_w100.tsv"
output_txt = "shard_docs.txt"

with open(input_tsv, "r", encoding="utf8") as fin, open(output_txt, "w", encoding="utf8") as fout:
    for i, line in enumerate(fin):
        parts = line.rstrip("\n").split("\t")
        # Use just the passage text, or (title + text)
        if len(parts) == 3:
            # This line saves just the text
            # fout.write(parts[2] + "\n")
            # If you want title + text (sometimes better for retrieval):
            fout.write(parts[1] + " " + parts[2] + "\n")
        else:
            continue  # skip bad lines
print(f"Done! {i+1} passages written to {output_txt}")
