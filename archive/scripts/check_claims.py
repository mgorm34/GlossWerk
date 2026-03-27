# Save as check_claims.py
import csv

with open(r"C:\glosswerk\data\raw\patent_claims_abstracts.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    print("Columns:", headers)
    
    row_count = 0
    has_claims = 0
    has_abstract = 0
    ipc_counts = {}
    
    for row in reader:
        row_count += 1
        if row.get("claims_de") and row.get("claims_en"):
            has_claims += 1
        if row.get("abstract_de") and row.get("abstract_en"):
            has_abstract += 1
        ipc = row.get("ipc_code", "")[:4]
        ipc_counts[ipc] = ipc_counts.get(ipc, 0) + 1
    
    print(f"\nTotal rows: {row_count}")
    print(f"With DE+EN claims: {has_claims}")
    print(f"With DE+EN abstracts: {has_abstract}")
    print(f"\nIPC distribution:")
    for ipc, count in sorted(ipc_counts.items(), key=lambda x: -x[1]):
        print(f"  {ipc}: {count}")