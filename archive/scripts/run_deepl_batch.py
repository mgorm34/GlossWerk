# Save as run_deepl_batch.py
import os
import deepl

auth_key = os.environ.get("DEEPL_AUTH_KEY")
if not auth_key:
    print("Set DEEPL_AUTH_KEY first!")
    exit()

translator = deepl.Translator(auth_key)

with open(r"C:\glosswerk\data\a61f_deepl_batch.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Translating {len(sentences)} sentences...")

translations = []
for i in range(0, len(sentences), 50):
    batch = sentences[i:i+50]
    results = translator.translate_text(batch, source_lang="DE", target_lang="EN-US")
    for r in results:
        translations.append(r.text)
    if (i + 50) % 500 == 0 or i + 50 >= len(sentences):
        print(f"  {min(i+50, len(sentences))}/{len(sentences)}")

with open(r"C:\glosswerk\data\a61f_deepl_translations.txt", "w", encoding="utf-8") as f:
    for t in translations:
        f.write(t + "\n")

# Calculate approximate cost
total_chars = sum(len(s) for s in sentences)
cost = total_chars / 1_000_000 * 20
print(f"\nDone. Total characters: {total_chars:,}")
print(f"Approximate cost: ${cost:.2f}")
print(f"Saved to a61f_deepl_translations.txt")
