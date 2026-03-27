"""Quick diagnostic: see what the Stage A model actually outputs."""
import torch
import sqlite3
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = r"C:\glosswerk\models\patent_ape_stageA\final"
db_path = r"C:\glosswerk\data\glosswerk_patent.db"

print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")
model.eval()

conn = sqlite3.connect(db_path)
rows = conn.execute("SELECT mt_opus, ref FROM sentence_pairs WHERE split = 'test' LIMIT 5").fetchall()
conn.close()

print(f"\nTesting {len(rows)} samples:\n")
for i, (mt, ref) in enumerate(rows):
    inp = tokenizer(f"postedit: {mt}", return_tensors="pt", max_length=256, truncation=True).to("cuda")
    out = model.generate(**inp, max_length=256, num_beams=4)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"--- Sample {i+1} ---")
    print(f"  INPUT:  {mt[:120]}")
    print(f"  OUTPUT: {decoded[:120]}")
    print(f"  REF:    {ref[:120]}")
    print()
