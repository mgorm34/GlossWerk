# Save as check_hter_db.py
import sqlite3

conn = sqlite3.connect(r"C:\glosswerk\data\hter_training\hter_training.db")

total = conn.execute("SELECT COUNT(*) FROM domain_pairs").fetchone()[0]
train = conn.execute("SELECT COUNT(*) FROM domain_pairs WHERE split='train'").fetchone()[0]
val = conn.execute("SELECT COUNT(*) FROM domain_pairs WHERE split='val'").fetchone()[0]
deepl = conn.execute("SELECT COUNT(*) FROM domain_pairs WHERE mt_deepl IS NOT NULL AND mt_deepl != ''").fetchone()[0]

print(f"Total: {total}")
print(f"Train: {train}")
print(f"Val: {val}")
print(f"With DeepL: {deepl}")

row = conn.execute("SELECT mt_deepl, ref FROM domain_pairs WHERE mt_deepl IS NOT NULL LIMIT 1").fetchone()
print(f"\nSample DeepL: {row[0][:100]}...")
print(f"Sample Correction: {row[1][:100]}...")

conn.close()