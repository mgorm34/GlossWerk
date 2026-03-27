# Save as extract_medical_pairs.py
import sqlite3

conn = sqlite3.connect(r"C:\glosswerk\data\glosswerk_patent.db")
out = sqlite3.connect(r"C:\glosswerk\data\medical_training.db")

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
table = tables[0][0]

medical_terms = ['stent', 'valve', 'catheter', 'implant', 'prosthe', 'surgical', 'cardiac', 'aortic', 'mitral', 'endoscop', 'cannula', 'graft', 'artery', 'ventricl', 'atrial', 'coronary', 'orthoped', 'hip joint', 'knee joint', 'bone', 'tissue', 'biocompat', 'suture']

out.execute("""
    CREATE TABLE IF NOT EXISTS domain_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        src TEXT NOT NULL,
        ref TEXT NOT NULL,
        mt_opus TEXT,
        mt_deepl TEXT,
        ipc_code TEXT DEFAULT 'A61_medical',
        publication TEXT,
        section TEXT DEFAULT 'description',
        split TEXT DEFAULT 'train'
    )
""")
out.execute("DELETE FROM domain_pairs")

medical_ids = set()
for term in medical_terms:
    rows = conn.execute(f"SELECT id FROM {table} WHERE ref LIKE ?", (f"%{term}%",)).fetchall()
    for r in rows:
        medical_ids.add(r[0])

print(f"Extracting {len(medical_ids)} medical pairs...")

import random
random.seed(42)
id_list = sorted(medical_ids)
random.shuffle(id_list)

n = len(id_list)
n_val = max(500, n // 20)
n_test = max(500, n // 20)

for i, row_id in enumerate(id_list):
    row = conn.execute(f"SELECT src, ref, mt_opus FROM {table} WHERE id=?", (row_id,)).fetchone()
    if not row:
        continue
    
    if i < n_val:
        split = "val"
    elif i < n_val + n_test:
        split = "test"
    else:
        split = "train"
    
    out.execute(
        "INSERT INTO domain_pairs (src, ref, mt_opus, ipc_code, split) VALUES (?,?,?,?,?)",
        (row[0], row[1], row[2], "A61_medical", split)
    )

out.commit()

for split in ["train", "val", "test"]:
    count = out.execute(f"SELECT COUNT(*) FROM domain_pairs WHERE split=?", (split,)).fetchone()[0]
    print(f"  {split}: {count}")

out.close()
conn.close()
print("\nSaved to C:\\glosswerk\\data\\medical_training.db")