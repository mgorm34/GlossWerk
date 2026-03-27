import sqlite3
conn = sqlite3.connect(r'C:\glosswerk\data\glosswerk_ape (Newest).db')
conn.row_factory = sqlite3.Row
c = conn.cursor()
rows = c.execute("SELECT src, mt_opus, ref, ter, domain FROM sentence_pairs WHERE ter <= 0.3 AND split = 'test' LIMIT 5").fetchall()
for r in rows:
    print(f"TER: {r['ter']:.2f} | Domain: {r['domain']}")
    print(f"  DE:  {r['src'][:100]}")
    print(f"  MT:  {r['mt_opus'][:100]}")
    print(f"  REF: {r['ref'][:100]}")
    print()