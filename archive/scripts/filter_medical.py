# Save as filter_medical.py
import sqlite3

conn = sqlite3.connect(r"C:\glosswerk\data\glosswerk_patent.db")

# Get the actual table name
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
table = tables[0][0]

medical_terms = ['stent', 'valve', 'catheter', 'implant', 'prosthe', 'surgical', 'cardiac', 'aortic', 'mitral', 'endoscop', 'cannula', 'graft', 'artery', 'ventricl', 'atrial', 'coronary', 'orthoped', 'hip joint', 'knee joint', 'bone', 'tissue', 'biocompat', 'suture']

total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
print(f"Total pairs in DB: {total}")

# Search for medical terms in the English reference
medical_ids = set()
for term in medical_terms:
    count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE ref LIKE ?", (f"%{term}%",)).fetchone()[0]
    if count > 0:
        print(f"  '{term}': {count} matches")
        rows = conn.execute(f"SELECT id FROM {table} WHERE ref LIKE ?", (f"%{term}%",)).fetchall()
        for r in rows:
            medical_ids.add(r[0])

print(f"\nTotal unique medical pairs: {len(medical_ids)}")
conn.close()