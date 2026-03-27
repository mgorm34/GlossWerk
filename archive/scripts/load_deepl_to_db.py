# Save as load_deepl_to_db.py
import sqlite3

conn = sqlite3.connect(r"C:\glosswerk\data\domain_patent_training.db")

# Add DeepL column if not exists
try:
    conn.execute("ALTER TABLE domain_pairs ADD COLUMN mt_deepl TEXT")
except:
    pass  # Column already exists

# Load IDs and translations
with open(r"C:\glosswerk\data\a61f_deepl_ids.txt", "r") as f:
    ids = [int(line.strip()) for line in f if line.strip()]

with open(r"C:\glosswerk\data\a61f_deepl_translations.txt", "r", encoding="utf-8") as f:
    translations = [line.strip() for line in f]

print(f"IDs: {len(ids)}, Translations: {len(translations)}")

if len(ids) != len(translations):
    print("ERROR: Mismatch!")
    exit()

for i, (row_id, trans) in enumerate(zip(ids, translations)):
    conn.execute("UPDATE domain_pairs SET mt_deepl=? WHERE id=?", (trans, row_id))

conn.commit()

# Verify
count = conn.execute("SELECT COUNT(*) FROM domain_pairs WHERE mt_deepl IS NOT NULL").fetchone()[0]
print(f"Rows with DeepL: {count}")

# Show a sample
row = conn.execute("SELECT src, mt_deepl, ref FROM domain_pairs WHERE mt_deepl IS NOT NULL LIMIT 1").fetchone()
print(f"\nDE:    {row[0][:100]}...")
print(f"DeepL: {row[1][:100]}...")
print(f"Ref:   {row[2][:100]}...")

conn.close()