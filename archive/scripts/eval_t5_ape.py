"""
GlossWerk APE - Model Evaluation Script
Runs the trained T5 model on test data and computes BLEU/chrF scores.

Usage:
    conda activate glosswerk
    python eval_t5_ape.py --db "C:\glosswerk\data\glosswerk_ape (Newest).db" --model "C:\glosswerk\models\t5_ape_model_v3\final"
"""

import argparse
import sqlite3
import time

import sacrebleu
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_test_data(db_path, limit=None):
    """Load test data from SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT mt_opus, ref, src, domain FROM sentence_pairs WHERE split = 'test'"
    if limit:
        query += f" LIMIT {limit}"

    rows = [dict(row) for row in cursor.execute(query).fetchall()]
    conn.close()
    return rows


def generate_corrections(model, tokenizer, mt_outputs, batch_size=32, max_length=256):
    """Run the model on MT outputs and return corrections."""
    device = model.device
    corrections = []

    for i in range(0, len(mt_outputs), batch_size):
        batch = mt_outputs[i : i + batch_size]
        inputs = [f"postedit: {text}" for text in batch]

        tokenized = tokenizer(
            inputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        corrections.extend(decoded)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(mt_outputs))}/{len(mt_outputs)}")

    return corrections


def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 APE model")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit test samples (for quick testing)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model from: {args.model}")
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()
    print("Model loaded.")

    # Load test data
    test_data = load_test_data(args.db, args.limit)
    print(f"Test samples: {len(test_data)}")

    mt_outputs = [row["mt_opus"] for row in test_data]
    references = [row["ref"] for row in test_data]

    # Generate corrections
    print("\nGenerating corrections...")
    start_time = time.time()
    corrections = generate_corrections(
        model, tokenizer, mt_outputs, args.batch_size
    )
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s ({len(corrections) / elapsed:.1f} sentences/sec)")

    # Compute scores: raw MT vs human reference
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Baseline: raw opus-mt output vs human reference
    baseline_bleu = sacrebleu.corpus_bleu(mt_outputs, [references])
    baseline_chrf = sacrebleu.corpus_chrf(mt_outputs, [references])

    # APE model: corrected output vs human reference
    ape_bleu = sacrebleu.corpus_bleu(corrections, [references])
    ape_chrf = sacrebleu.corpus_chrf(corrections, [references])

    print(f"\n{'Metric':<15} {'opus-mt (baseline)':<25} {'T5-base APE':<25} {'Improvement'}")
    print("-" * 80)
    print(
        f"{'BLEU':<15} {baseline_bleu.score:<25.2f} {ape_bleu.score:<25.2f} {ape_bleu.score - baseline_bleu.score:+.2f}"
    )
    print(
        f"{'chrF':<15} {baseline_chrf.score:<25.2f} {ape_chrf.score:<25.2f} {ape_chrf.score - baseline_chrf.score:+.2f}"
    )

    # Show some examples
    print(f"\n{'=' * 60}")
    print("SAMPLE CORRECTIONS (first 5)")
    print("=" * 60)
    for i in range(min(5, len(test_data))):
        print(f"\n--- Example {i + 1} (domain: {test_data[i]['domain']}) ---")
        print(f"  Source DE:    {test_data[i]['src'][:100]}...")
        print(f"  opus-mt:     {mt_outputs[i][:100]}...")
        print(f"  T5 APE:      {corrections[i][:100]}...")
        print(f"  Human ref:   {references[i][:100]}...")

    # Per-domain breakdown
    domains = set(row["domain"] for row in test_data)
    if len(domains) > 1:
        print(f"\n{'=' * 60}")
        print("PER-DOMAIN BREAKDOWN")
        print("=" * 60)
        print(f"\n{'Domain':<20} {'Baseline BLEU':<18} {'APE BLEU':<18} {'Improvement'}")
        print("-" * 70)

        for domain in sorted(domains):
            idx = [j for j, row in enumerate(test_data) if row["domain"] == domain]
            d_mt = [mt_outputs[j] for j in idx]
            d_corrections = [corrections[j] for j in idx]
            d_refs = [references[j] for j in idx]

            d_baseline = sacrebleu.corpus_bleu(d_mt, [d_refs])
            d_ape = sacrebleu.corpus_bleu(d_corrections, [d_refs])

            print(
                f"{domain:<20} {d_baseline.score:<18.2f} {d_ape.score:<18.2f} {d_ape.score - d_baseline.score:+.2f}"
            )

    print(f"\nEvaluation complete.")


if __name__ == "__main__":
    main()
