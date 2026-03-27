"""Diagnose why T5 training produces garbage output.
Checks: tokenization, data format, loss computation, single-step training."""
import torch
import sqlite3
from transformers import T5ForConditionalGeneration, T5Tokenizer

db_path = r"C:\glosswerk\data\glosswerk_patent.db"

# Load fresh T5-base (not our trained one)
print("Loading fresh T5-base...")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base").to("cuda")

# Get a sample from our DB
conn = sqlite3.connect(db_path)
row = conn.execute("SELECT mt_opus, ref FROM sentence_pairs WHERE split = 'train' LIMIT 1").fetchone()
conn.close()

mt_text = row[0]
ref_text = row[1]

print(f"\nSample data:")
print(f"  MT:  {mt_text[:100]}")
print(f"  REF: {ref_text[:100]}")

# Test 1: Check tokenization
input_text = f"postedit: {mt_text}"
print(f"\n--- Test 1: Tokenization ---")
input_enc = tokenizer(input_text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
target_enc = tokenizer(ref_text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")

print(f"  Input tokens (first 20):  {input_enc['input_ids'][0][:20].tolist()}")
print(f"  Target tokens (first 20): {target_enc['input_ids'][0][:20].tolist()}")
print(f"  Input decoded back: {tokenizer.decode(input_enc['input_ids'][0], skip_special_tokens=True)[:80]}")
print(f"  Target decoded back: {tokenizer.decode(target_enc['input_ids'][0], skip_special_tokens=True)[:80]}")

# Test 2: Check labels (are we masking correctly?)
print(f"\n--- Test 2: Label masking ---")
labels = target_enc["input_ids"].clone()
pad_count = (labels == tokenizer.pad_token_id).sum().item()
labels[labels == tokenizer.pad_token_id] = -100
masked_count = (labels == -100).sum().item()
real_token_count = (labels != -100).sum().item()
print(f"  Total tokens: {labels.shape[1]}")
print(f"  Pad tokens masked to -100: {masked_count}")
print(f"  Real target tokens: {real_token_count}")
print(f"  First 20 labels: {labels[0][:20].tolist()}")

# Test 3: Forward pass - does loss compute?
print(f"\n--- Test 3: Forward pass ---")
model.train()
outputs = model(
    input_ids=input_enc["input_ids"].to("cuda"),
    attention_mask=input_enc["attention_mask"].to("cuda"),
    labels=labels.to("cuda"),
)
print(f"  Loss: {outputs.loss.item():.4f}")

# Test 4: Quick 50-step training on 10 examples, then check output
print(f"\n--- Test 4: Quick training test (50 steps on 10 examples) ---")
conn = sqlite3.connect(db_path)
rows = conn.execute("SELECT mt_opus, ref FROM sentence_pairs WHERE split = 'train' LIMIT 10").fetchall()
conn.close()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(50):
    total_loss = 0
    for mt, ref in rows:
        inp = tokenizer(f"postedit: {mt}", max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        tgt = tokenizer(ref, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        lab = tgt["input_ids"].clone()
        lab[lab == tokenizer.pad_token_id] = -100

        out = model(
            input_ids=inp["input_ids"].to("cuda"),
            attention_mask=inp["attention_mask"].to("cuda"),
            labels=lab.to("cuda"),
        )
        out.loss.backward()
        total_loss += out.loss.item()

    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        avg = total_loss / len(rows)
        print(f"  Step {step}: avg_loss = {avg:.4f}")

# Test 5: Generate from quick-trained model
print(f"\n--- Test 5: Output from quick-trained model ---")
model.eval()
test_mt = rows[0][0]
test_ref = rows[0][1]
inp = tokenizer(f"postedit: {test_mt}", return_tensors="pt", max_length=256, truncation=True).to("cuda")

with torch.no_grad():
    out = model.generate(**inp, max_length=256, num_beams=4)
decoded = tokenizer.decode(out[0], skip_special_tokens=True)

print(f"  INPUT:  {test_mt[:120]}")
print(f"  OUTPUT: {decoded[:120]}")
print(f"  REF:    {test_ref[:120]}")

# Also try without prefix
inp2 = tokenizer(test_mt, return_tensors="pt", max_length=256, truncation=True).to("cuda")
with torch.no_grad():
    out2 = model.generate(**inp2, max_length=256, num_beams=4)
decoded2 = tokenizer.decode(out2[0], skip_special_tokens=True)
print(f"\n  (No prefix) OUTPUT: {decoded2[:120]}")
