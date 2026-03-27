import sqlite3
conn = sqlite3.connect(r"C:\glosswerk\data\glosswerk_patent.db")

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
table = tables[0][0]
print(f"Table: {table}")

print("\nDomains:")
for row in conn.execute(f"SELECT domain, COUNT(*) FROM {table} GROUP BY domain").fetchall():
    print(f"  {row[0]}: {row[1]}")

print("\nCorpus values:")
for row in conn.execute(f"SELECT corpus, COUNT(*) FROM {table} GROUP BY corpus").fetchall():
    print(f"  {row[0]}: {row[1]}")

print("\nSample row:")
row = conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
for i, col in enumerate(['id','src','ref','mt_opus','mt_deepl','ter','domain','corpus','split']):
    val = str(row[i])[:100] if row[i] else 'NULL'
    print(f"  {col}: {val}")