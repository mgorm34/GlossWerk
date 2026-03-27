# Save as pick_deepl_batch.py
import sqlite3
import random

conn = sqlite3.connect(r"C:\glosswerk\data\domain_patent_training.db")

# How many A61F sentences do we have?
total = conn.execute("SELECT COUNT(*) FROM domain_pairs WHERE ipc_code LIKE 'A61F%'").fetchone()[0]
print(f"Total A61F sentences: {total}")

# Pick 5000 random training sentences from A61F
rows = conn.execute(
    "SELECT id, src, ref FROM domain_pairs WHERE ipc_code LIKE 'A61F%' AND split='train' ORDER BY RANDOM() LIMIT 5000"
).fetchall()

print(f"Selected {len(rows)} sentences for DeepL translation")

# Save the German sentences to a text file for DeepL
with open(r"C:\glosswerk\data\a61f_deepl_batch.txt", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(row[1] + "\n")

# Save the IDs so we can match back later
with open(r"C:\glosswerk\data\a61f_deepl_ids.txt", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(str(row[0]) + "\n")

print(f"Saved to a61f_deepl_batch.txt")
print(f"Sample sentence: {rows[0][1][:100]}...")

conn.close()