"""
Step 2: Build domain-specific training database from downloaded corpora.
Processes any corpus into the standard GlossWerk format:
  - Loads DE-EN pairs with quality filtering
  - Runs opus-mt for baseline MT
  - Computes TER
  - Filters and splits train/val/test
  - Saves to SQLite

Usage:
    # Build patent database (start here):
    python 02_build_domain_db.py --corpus europat --domain patent

    # Build medical database:
    python 02_build_domain_db.py --corpus emea --domain medical

    # Build legal database:
    python 02_build_domain_db.py --corpus dgt --domain legal

    # Build from multiple corpora into one DB:
    python 02_build_domain_db.py --corpus europat emea dgt --domain multi

    # Limit pairs for quick testing:
    python 02_build_domain_db.py --corpus europat --domain patent --max_pairs 10000
"""

import argparse
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter

import torch
from transformers import MarianMTModel, MarianTokenizer


def load_parallel_data(data_dir, corpus_name, max_pairs=None, skip_lines=0):
    """Load parallel sentence pairs from Moses-format files."""
    corpus_dir = os.path.join(data_dir, corpus_name)

    if not os.path.exists(corpus_dir):
        print(f"  ERROR: Directory not found: {corpus_dir}")
        print(f"  Run 01_download_europat.py --only {corpus_name} first.")
        return []

    de_files = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".de")])
    en_files = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".en")])

    if not de_files or not en_files:
        print(f"  ERROR: No .de/.en files in {corpus_dir}")
        return []

    de_path = os.path.join(corpus_dir, de_files[0])
    en_path = os.path.join(corpus_dir, en_files[0])

    print(f"  Loading: {de_path}")
    print(f"           {en_path}")
    if skip_lines > 0:
        print(f"  Skipping first {skip_lines:,} lines")

    pairs = []
    skipped = Counter()

    with open(de_path, "r", encoding="utf-8", errors="replace") as f_de, \
         open(en_path, "r", encoding="utf-8", errors="replace") as f_en:
        for i, (de_line, en_line) in enumerate(zip(f_de, f_en)):
            if i < skip_lines:
                continue
            if max_pairs and len(pairs) >= max_pairs:
                break

            src = de_line.strip()
            ref = en_line.strip()

            # Quality filters
            if not src or not ref:
                skipped["empty"] += 1
                continue
            if len(src) < 10 or len(ref) < 10:
                skipped["too_short"] += 1
                continue
            if len(src) > 500 or len(ref) > 500:
                skipped["too_long"] += 1
                continue
            if len(re.findall(r"[a-zA-ZäöüÄÖÜß]", src)) < len(src) * 0.4:
                skipped["mostly_numbers"] += 1
                continue
            if src.lower().strip() == ref.lower().strip():
                skipped["identical"] += 1
                continue
            # Skip very unbalanced pairs (likely misaligned)
            ratio = len(src) / max(len(ref), 1)
            if ratio > 3.0 or ratio < 0.3:
                skipped["length_ratio"] += 1
                continue

            pairs.append({"src": src, "ref": ref, "corpus": corpus_name})

            if (i + 1) % 500000 == 0:
                print(f"  Read {i + 1:,} lines, kept {len(pairs):,}")

    print(f"  Loaded {len(pairs):,} pairs from {corpus_name}")
    if skipped:
        print(f"  Filtered out: {dict(skipped)}")
    return pairs


def translate_batch(model, tokenizer, texts, device, batch_size=64):
    """Translate German texts to English using opus-mt."""
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=4)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

        if (i // batch_size) % 100 == 0 and i > 0:
            speed = len(translations) / max(1, time.time() - translate_batch._start)
            print(f"  Translated {len(translations):,}/{len(texts):,} ({speed:.0f} sent/sec)")

    return translations


def compute_ter(hypothesis, reference):
    """Word-level Translation Error Rate."""
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    if not ref_words:
        return 1.0

    m, n = len(hyp_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Build domain training database")
    parser.add_argument("--corpus", nargs="+", required=True,
                        help="Corpus name(s) to process (e.g., europat emea dgt)")
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain label (e.g., patent, medical, legal, multi)")
    default_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--output_db", type=str, default=None,
                        help="Output DB path (default: data/glosswerk_{domain}.db)")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max pairs per corpus (for quick testing)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N lines of corpus (for chunked processing)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing DB instead of overwriting")
    parser.add_argument("--ter_threshold", type=float, default=0.5,
                        help="Maximum TER for clean pairs (default: 0.5)")
    args = parser.parse_args()

    if not args.output_db:
        default_db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(default_db_dir, exist_ok=True)
        args.output_db = os.path.join(default_db_dir, f"glosswerk_{args.domain}.db")

    print("=" * 60)
    print(f"GlossWerk - {args.domain.title()} Database Builder")
    print("=" * 60)

    # Load all corpora
    all_pairs = []
    for corpus_name in args.corpus:
        print(f"\n--- Loading {corpus_name} ---")
        pairs = load_parallel_data(args.data_dir, corpus_name, args.max_pairs, args.skip)
        all_pairs.extend(pairs)

    if not all_pairs:
        print("ERROR: No data loaded. Check corpus names and run 01_download_europat.py first.")
        sys.exit(1)

    print(f"\nTotal pairs loaded: {len(all_pairs):,}")

    # Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load opus-mt
    print("\n--- Loading opus-mt-de-en ---")
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    # Translate
    print("\n--- Translating with opus-mt ---")
    src_texts = [p["src"] for p in all_pairs]
    translate_batch._start = time.time()
    mt_outputs = translate_batch(model, tokenizer, src_texts, device)
    elapsed = time.time() - translate_batch._start
    print(f"  Done: {elapsed:.0f}s ({len(all_pairs) / elapsed:.0f} sent/sec)")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Compute TER
    print("\n--- Computing TER ---")
    for i, pair in enumerate(all_pairs):
        pair["mt_opus"] = mt_outputs[i]
        pair["ter"] = compute_ter(mt_outputs[i], pair["ref"])

        if (i + 1) % 200000 == 0:
            print(f"  TER computed for {i + 1:,}/{len(all_pairs):,}")

    # TER distribution
    ter_buckets = Counter()
    for p in all_pairs:
        ter_buckets[round(p["ter"], 1)] += 1

    print("\n  TER Distribution:")
    for bucket in sorted(ter_buckets.keys()):
        count = ter_buckets[bucket]
        bar = "#" * min(50, count // max(1, max(ter_buckets.values()) // 50))
        print(f"    {bucket:.1f}: {count:>8,} {bar}")

    # Filter
    clean_pairs = [p for p in all_pairs if p["ter"] <= args.ter_threshold]
    print(f"\n  Total pairs: {len(all_pairs):,}")
    print(f"  Clean pairs (TER <= {args.ter_threshold}): {len(clean_pairs):,}")
    print(f"  Filtered out: {len(all_pairs) - len(clean_pairs):,}")

    if not clean_pairs:
        print("ERROR: No pairs survived TER filtering. Try a higher --ter_threshold.")
        sys.exit(1)

    # Split
    random.seed(42)
    random.shuffle(clean_pairs)
    n = len(clean_pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    for i, pair in enumerate(clean_pairs):
        if i < train_end:
            pair["split"] = "train"
        elif i < val_end:
            pair["split"] = "val"
        else:
            pair["split"] = "test"

    split_counts = Counter(p["split"] for p in clean_pairs)
    corpus_counts = Counter(p["corpus"] for p in clean_pairs)

    print(f"\n  Splits: {dict(split_counts)}")
    print(f"  By corpus: {dict(corpus_counts)}")

    # Save to SQLite
    print(f"\n--- Saving to {args.output_db} ---")
    if args.append and os.path.exists(args.output_db):
        print(f"  APPEND MODE: Adding to existing database")
        conn = sqlite3.connect(args.output_db)
        cursor = conn.cursor()
        existing = cursor.execute("SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
        print(f"  Existing pairs: {existing:,}")
    else:
        if os.path.exists(args.output_db):
            os.remove(args.output_db)
        conn = sqlite3.connect(args.output_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE sentence_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT NOT NULL,
                ref TEXT NOT NULL,
                mt_opus TEXT NOT NULL,
                mt_deepl TEXT,
                ter REAL NOT NULL,
                domain TEXT NOT NULL,
                corpus TEXT NOT NULL,
                split TEXT NOT NULL
            )
        """)

    cursor.executemany(
        "INSERT INTO sentence_pairs (src, ref, mt_opus, ter, domain, corpus, split) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(p["src"], p["ref"], p["mt_opus"], p["ter"], args.domain, p["corpus"], p["split"])
         for p in clean_pairs],
    )
    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
    print(f"  Saved {count:,} pairs")
    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Database ready: {args.output_db}")
    print(f"\nNext steps:")
    print(f"  1. Train: python 03_train_patent_model.py --stage A --db \"{args.output_db}\"")
    print(f"  2. DeepL: python 04_deepl_corrections.py --db \"{args.output_db}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
