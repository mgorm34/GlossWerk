"""
Process BigQuery patent abstracts into APE training data.

Steps:
  1. Load CSV from BigQuery export
  2. Sentence-split DE and EN abstracts
  3. Align sentences by position
  4. Translate DE sentences with opus-mt to get MT baseline
  5. Save as training DB: (source_de, mt_opus, reference_en, ipc_code)

Usage:
  python 14_process_bigquery_data.py
  python 14_process_bigquery_data.py --ipc A61F  (filter to specific domain)
  python 14_process_bigquery_data.py --skip_mt  (just split and align, add MT later)
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import Counter

import torch
from transformers import MarianMTModel, MarianTokenizer


def split_sentences(text):
    """Split patent text into sentences."""
    if not text or not text.strip():
        return []
    # Clean up common patent artifacts
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Split on period/question/exclamation followed by space and capital letter
    # But not on common abbreviations like e.g., i.e., etc., Fig., No.
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ\[(])', text)
    
    sentences = []
    for part in parts:
        part = part.strip()
        if len(part) > 15:  # Skip very short fragments
            sentences.append(part)
    return sentences


def align_sentences(de_sents, en_sents):
    """Simple 1:1 alignment by position. Only return pairs where counts match."""
    if len(de_sents) == len(en_sents) and len(de_sents) > 0:
        return list(zip(de_sents, en_sents))
    
    # If counts don't match, try to salvage what we can
    # Only take pairs up to the shorter list
    if len(de_sents) > 0 and len(en_sents) > 0:
        min_len = min(len(de_sents), len(en_sents))
        # Only use if they're close in count (within 1)
        if abs(len(de_sents) - len(en_sents)) <= 1:
            return list(zip(de_sents[:min_len], en_sents[:min_len]))
    
    return []


def translate_batch_opus(sentences, model, tokenizer, device, batch_size=32):
    """Translate DE sentences to EN using opus-mt."""
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                          truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
        
        if (i + batch_size) % 1000 < batch_size:
            print(f"    Translated {min(i+batch_size, len(sentences))}/{len(sentences)}")
    
    return translations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"C:\glosswerk\data\raw\patent_claims_abstracts.csv")
    parser.add_argument("--output_db", default=r"C:\glosswerk\data\domain_patent_training.db")
    parser.add_argument("--ipc", default=None, help="Filter to specific IPC prefix, e.g. A61F")
    parser.add_argument("--skip_mt", action="store_true", help="Skip MT translation step")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_patents", type=int, default=None, help="Limit number of patents")
    args = parser.parse_args()

    # --- 1. Load and parse CSV ---
    print("--- Step 1: Load CSV ---")
    patents = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ipc = row.get("ipc_code", "")
            if args.ipc and not ipc.startswith(args.ipc):
                continue
            
            abstract_de = row.get("abstract_de", "")
            abstract_en = row.get("abstract_en", "")
            claims_de = row.get("claims_de", "")
            claims_en = row.get("claims_en", "")
            
            if abstract_de and abstract_en:
                patents.append({
                    "pub": row.get("publication_number", ""),
                    "family_id": row.get("family_id", ""),
                    "ipc": ipc,
                    "abstract_de": abstract_de,
                    "abstract_en": abstract_en,
                    "claims_de": claims_de or "",
                    "claims_en": claims_en or "",
                })
            
            if args.max_patents and len(patents) >= args.max_patents:
                break
    
    print(f"  Loaded {len(patents)} patents")
    
    # IPC distribution
    ipc_counts = Counter(p["ipc"][:4] for p in patents)
    print(f"  IPC distribution (top 10):")
    for ipc, count in ipc_counts.most_common(10):
        print(f"    {ipc}: {count}")

    # --- 2. Sentence split and align ---
    print("\n--- Step 2: Sentence split and align ---")
    pairs = []
    aligned_patents = 0
    
    for patent in patents:
        # Process abstracts
        de_sents = split_sentences(patent["abstract_de"])
        en_sents = split_sentences(patent["abstract_en"])
        aligned = align_sentences(de_sents, en_sents)
        
        if aligned:
            aligned_patents += 1
            for de, en in aligned:
                pairs.append({
                    "src": de,
                    "ref": en,
                    "ipc": patent["ipc"],
                    "pub": patent["pub"],
                    "section": "abstract",
                })
        
        # Process claims if available
        if patent["claims_de"] and patent["claims_en"]:
            de_claims = split_sentences(patent["claims_de"])
            en_claims = split_sentences(patent["claims_en"])
            claim_aligned = align_sentences(de_claims, en_claims)
            
            if claim_aligned:
                for de, en in claim_aligned:
                    pairs.append({
                        "src": de,
                        "ref": en,
                        "ipc": patent["ipc"],
                        "pub": patent["pub"],
                        "section": "claims",
                    })
    
    print(f"  Aligned patents: {aligned_patents}/{len(patents)}")
    print(f"  Total sentence pairs: {len(pairs)}")
    
    section_counts = Counter(p["section"] for p in pairs)
    print(f"  From abstracts: {section_counts.get('abstract', 0)}")
    print(f"  From claims: {section_counts.get('claims', 0)}")

    if not pairs:
        print("ERROR: No aligned pairs found!")
        sys.exit(1)

    # --- 3. Generate MT with opus-mt ---
    if not args.skip_mt:
        print("\n--- Step 3: Generate opus-mt translations ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading opus-mt on {device}...")
        
        mt_model_name = "Helsinki-NLP/opus-mt-de-en"
        mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        mt_model = MarianMTModel.from_pretrained(mt_model_name).to(device)
        mt_model.eval()
        
        de_sentences = [p["src"] for p in pairs]
        mt_translations = translate_batch_opus(
            de_sentences, mt_model, mt_tokenizer, device, args.batch_size
        )
        
        for i, mt in enumerate(mt_translations):
            pairs[i]["mt_opus"] = mt
        
        del mt_model, mt_tokenizer
        torch.cuda.empty_cache()
        print(f"  Generated {len(mt_translations)} translations")
    else:
        for p in pairs:
            p["mt_opus"] = ""
        print("\n--- Step 3: Skipped MT (--skip_mt) ---")

    # --- 4. Save to SQLite ---
    print(f"\n--- Step 4: Save to {args.output_db} ---")
    
    conn = sqlite3.connect(args.output_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src TEXT NOT NULL,
            ref TEXT NOT NULL,
            mt_opus TEXT,
            ipc_code TEXT,
            publication TEXT,
            section TEXT,
            split TEXT DEFAULT 'train'
        )
    """)
    conn.execute("DELETE FROM domain_pairs")  # Clear existing data
    
    # 90/5/5 split
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    n = len(pairs)
    n_val = max(500, n // 20)
    n_test = max(500, n // 20)
    
    for i, p in enumerate(pairs):
        if i < n_val:
            split = "val"
        elif i < n_val + n_test:
            split = "test"
        else:
            split = "train"
        
        conn.execute(
            "INSERT INTO domain_pairs (src, ref, mt_opus, ipc_code, publication, section, split) VALUES (?,?,?,?,?,?,?)",
            (p["src"], p["ref"], p.get("mt_opus", ""), p["ipc"], p["pub"], p["section"], split)
        )
    
    conn.commit()
    
    # Summary
    for split in ["train", "val", "test"]:
        count = conn.execute(f"SELECT COUNT(*) FROM domain_pairs WHERE split=?", (split,)).fetchone()[0]
        print(f"  {split}: {count}")
    
    # IPC breakdown in training set
    print(f"\n  IPC distribution in training set:")
    rows = conn.execute(
        "SELECT substr(ipc_code,1,4) as ipc, COUNT(*) as cnt FROM domain_pairs WHERE split='train' GROUP BY ipc ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    for ipc, cnt in rows:
        print(f"    {ipc}: {cnt}")
    
    conn.close()
    print(f"\n  Saved to {args.output_db}")
    print("  Ready for training!")


if __name__ == "__main__":
    main()
