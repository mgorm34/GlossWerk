import sqlite3
conn = sqlite3.connect(r"C:\glosswerk\data\glosswerk_patent.db")
for split in ["test", "val", "train"]:
    r = conn.execute("SELECT COUNT(*), SUM(LENGTH(src)) FROM sentence_pairs WHERE split = ?", (split,)).fetchone()
    print(f"{split:6s}: {r[0]:>8,} sentences, {r[1]:>12,} characters")
conn.close()
