"""
Step 4: Generate DeepL translations for patent sentences.
Uses DeepL API Free tier (500k chars/month) to translate a sample
of patent source sentences, then stores them alongside the human
references for Stage B fine-tuning.

Setup:
    1. Sign up at https://www.deepl.com/pro-api (free tier)
    2. Get your API key from Account Settings
    3. pip install deepl --break-system-packages

Usage:
    conda activate glosswerk
    cd C:\glosswerk\scripts
    python 04_deepl_corrections.py --api_key YOUR_DEEPL_API_KEY

    # Or set as environment variable:
    set DEEPL_AUTH_KEY=YOUR_DEEPL_API_KEY
    python 04_deepl_corrections.py
"""

import argparse
import os
import sqlite3
import sys
import time

try:
    import deepl
except ImportError:
    print("Installing deepl package...")
    os.system("pip install deepl")
    import deepl


def get_char_budget(translator):
    """Check remaining character budget."""
    usage = translator.get_usage()
    if usage.character:
        used = usage.character.count
        limit = usage.character.limit
        remaining = limit - used
        print(f"  Characters used: {used:,} / {limit:,}")
        print(f"  Remaining: {remaining:,}")
        return remaining
    return 500000  # Assume free tier default


def estimate_chars(texts):
    """Estimate total characters in a list of texts."""
    return sum(len(t) for t in texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None,
                        help="DeepL API key (or set DEEPL_AUTH_KEY env var)")
    parser.add_argument("--db", type=str, default=r"C:\glosswerk\data\glosswerk_patent.db")
    parser.add_argument("--max_chars", type=int, default=400000,
                        help="Max characters to translate (leave buffer for free tier)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Sentences per API call")
    parser.add_argument("--priority", type=str, default="test",
                        choices=["test", "train", "all"],
                        help="Which split to prioritize translating")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("DEEPL_AUTH_KEY")
    if not api_key:
        print("ERROR: No DeepL API key provided.")
        print("  Option 1: python 04_deepl_corrections.py --api_key YOUR_KEY")
        print("  Option 2: set DEEPL_AUTH_KEY=YOUR_KEY")
        print("\n  Sign up free at: https://www.deepl.com/pro-api")
        sys.exit(1)

    print("=" * 60)
    print("GlossWerk - DeepL Translation Generator")
    print("=" * 60)

    # Initialize DeepL
    translator = deepl.Translator(api_key)
    remaining_chars = get_char_budget(translator)
    budget = min(args.max_chars, remaining_chars)
    print(f"  Translation budget: {budget:,} characters")

    # Add mt_deepl column if it doesn't exist
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE sentence_pairs ADD COLUMN mt_deepl TEXT")
        conn.commit()
        print("  Added mt_deepl column to database")
    except sqlite3.OperationalError:
        print("  mt_deepl column already exists")

    # Load sentences to translate (prioritize test set, then val, then train)
    if args.priority == "test":
        splits = ["test", "val", "train"]
    elif args.priority == "train":
        splits = ["train", "val", "test"]
    else:
        splits = ["test", "val", "train"]

    total_translated = 0
    chars_used = 0

    for split in splits:
        if chars_used >= budget:
            break

        rows = cursor.execute(
            "SELECT id, src FROM sentence_pairs WHERE split = ? AND mt_deepl IS NULL ORDER BY RANDOM()",
            (split,),
        ).fetchall()

        if not rows:
            print(f"\n  {split}: all sentences already translated")
            continue

        print(f"\n--- Translating {split} set ({len(rows):,} remaining) ---")

        batch = []
        batch_ids = []

        for row_id, src_text in rows:
            if chars_used + len(src_text) > budget:
                print(f"  Approaching character budget, stopping.")
                break

            batch.append(src_text)
            batch_ids.append(row_id)

            if len(batch) >= args.batch_size:
                # Translate batch
                try:
                    results = translator.translate_text(
                        batch, source_lang="DE", target_lang="EN-US"
                    )

                    for i, result in enumerate(results):
                        cursor.execute(
                            "UPDATE sentence_pairs SET mt_deepl = ? WHERE id = ?",
                            (result.text, batch_ids[i]),
                        )

                    chars_used += sum(len(t) for t in batch)
                    total_translated += len(batch)
                    conn.commit()

                    print(f"  Translated {total_translated:,} sentences ({chars_used:,} chars used)")

                except deepl.exceptions.QuotaExceededException:
                    print("  DeepL quota exceeded! Saving progress.")
                    conn.commit()
                    break
                except Exception as e:
                    print(f"  Error: {e}")
                    time.sleep(2)
                    continue

                batch = []
                batch_ids = []

                # Rate limiting - be gentle with free tier
                time.sleep(0.5)

        # Process remaining batch
        if batch:
            try:
                results = translator.translate_text(batch, source_lang="DE", target_lang="EN-US")
                for i, result in enumerate(results):
                    cursor.execute(
                        "UPDATE sentence_pairs SET mt_deepl = ? WHERE id = ?",
                        (result.text, batch_ids[i]),
                    )
                chars_used += sum(len(t) for t in batch)
                total_translated += len(batch)
                conn.commit()
            except Exception as e:
                print(f"  Error on final batch: {e}")

    # Summary
    deepl_count = cursor.execute(
        "SELECT COUNT(*) FROM sentence_pairs WHERE mt_deepl IS NOT NULL"
    ).fetchone()[0]

    deepl_by_split = cursor.execute(
        "SELECT split, COUNT(*) FROM sentence_pairs WHERE mt_deepl IS NOT NULL GROUP BY split"
    ).fetchall()

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"DeepL translation complete!")
    print(f"  Total translated this run: {total_translated:,}")
    print(f"  Characters used: {chars_used:,}")
    print(f"  Total with DeepL translations: {deepl_count:,}")
    for split, count in deepl_by_split:
        print(f"    {split}: {count:,}")
    print(f"\nNext step: python 03_train_patent_model.py --stage B")
    print("=" * 60)


if __name__ == "__main__":
    main()
