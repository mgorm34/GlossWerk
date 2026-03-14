"""
Step 5: Comprehensive evaluation of APE model.
Compares: raw opus-mt | raw DeepL | DeepL + APE correction
Shows overall scores, per-domain breakdown, and sample corrections.

Usage:
    # Evaluate Stage A model (opus-mt corrections only):
    python 05_evaluate.py --model ../models/patent_ape_stageA/final

    # Evaluate Stage B model (DeepL fine-tuned):
    python 05_evaluate.py --model ../models/patent_ape_stageB/final

    # Quick test with fewer samples:
    python 05_evaluate.py --model ../models/patent_ape_stageA/final --limit 200
"""

import argparse
import sqlite3
import time
import json
import os
from datetime import datetime

import sacrebleu
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_test_data(db_path, limit=None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT src, mt_opus, ref, domain, mt_deepl FROM sentence_pairs WHERE split = 'test'"
    if limit:
        query += f" LIMIT {limit}"

    rows = [dict(row) for row in cursor.execute(query).fetchall()]
    conn.close()
    return rows


def generate_corrections(model, tokenizer, texts, device, batch_size=32, max_length=256):
    corrections = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = [f"postedit: {t}" for t in batch]

        tokenized = tokenizer(
            inputs, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized, max_length=max_length, num_beams=4, early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        corrections.extend(decoded)

        if (i // batch_size) % 20 == 0 and i > 0:
            print(f"  {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    return corrections


def score(hypotheses, references):
    """Compute BLEU and chrF."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_db = os.path.join(project_root, "data", "glosswerk_patent.db")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--db", type=str, default=default_db)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GlossWerk APE - Comprehensive Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    # Load test data
    test_data = load_test_data(args.db, args.limit)
    print(f"Test samples: {len(test_data):,}")

    references = [r["ref"] for r in test_data]
    mt_opus = [r["mt_opus"] for r in test_data]
    has_deepl = any(r["mt_deepl"] for r in test_data)

    # Score raw opus-mt
    print("\n--- Scoring raw opus-mt ---")
    opus_scores = score(mt_opus, references)

    # Correct opus-mt output with APE model
    print("\n--- Generating APE corrections on opus-mt ---")
    start = time.time()
    ape_opus = generate_corrections(model, tokenizer, mt_opus, device, args.batch_size)
    t_opus = time.time() - start
    ape_opus_scores = score(ape_opus, references)

    # If DeepL translations exist, score those too
    deepl_scores = None
    ape_deepl_scores = None
    if has_deepl:
        mt_deepl = [r["mt_deepl"] if r["mt_deepl"] else r["mt_opus"] for r in test_data]
        deepl_idx = [i for i, r in enumerate(test_data) if r["mt_deepl"]]

        if deepl_idx:
            dl_hyps = [mt_deepl[i] for i in deepl_idx]
            dl_refs = [references[i] for i in deepl_idx]

            print("\n--- Scoring raw DeepL ---")
            deepl_scores = score(dl_hyps, dl_refs)

            print("\n--- Generating APE corrections on DeepL ---")
            start = time.time()
            ape_deepl_all = generate_corrections(model, tokenizer, mt_deepl, device, args.batch_size)
            t_deepl = time.time() - start

            ape_dl_hyps = [ape_deepl_all[i] for i in deepl_idx]
            ape_deepl_scores = score(ape_dl_hyps, dl_refs)

    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'System':<30} {'BLEU':>10} {'chrF':>10}")
    print("-" * 50)
    print(f"{'opus-mt (baseline)':<30} {opus_scores['bleu']:>10.2f} {opus_scores['chrf']:>10.2f}")
    print(f"{'opus-mt + APE':<30} {ape_opus_scores['bleu']:>10.2f} {ape_opus_scores['chrf']:>10.2f}")
    delta_bleu = ape_opus_scores['bleu'] - opus_scores['bleu']
    delta_chrf = ape_opus_scores['chrf'] - opus_scores['chrf']
    print(f"{'  improvement':<30} {delta_bleu:>+10.2f} {delta_chrf:>+10.2f}")

    if deepl_scores and ape_deepl_scores:
        print()
        print(f"{'DeepL (baseline)':<30} {deepl_scores['bleu']:>10.2f} {deepl_scores['chrf']:>10.2f}")
        print(f"{'DeepL + APE':<30} {ape_deepl_scores['bleu']:>10.2f} {ape_deepl_scores['chrf']:>10.2f}")
        delta_bleu_dl = ape_deepl_scores['bleu'] - deepl_scores['bleu']
        delta_chrf_dl = ape_deepl_scores['chrf'] - deepl_scores['chrf']
        print(f"{'  improvement':<30} {delta_bleu_dl:>+10.2f} {delta_chrf_dl:>+10.2f}")

    # Per-domain breakdown
    domains = sorted(set(r["domain"] for r in test_data))
    if len(domains) > 1:
        print(f"\n{'=' * 70}")
        print("PER-DOMAIN BREAKDOWN")
        print(f"{'=' * 70}")
        print(f"\n{'Domain':<20} {'opus-mt BLEU':>15} {'APE BLEU':>15} {'Delta':>10}")
        print("-" * 60)

        for domain in domains:
            idx = [i for i, r in enumerate(test_data) if r["domain"] == domain]
            d_mt = [mt_opus[i] for i in idx]
            d_ape = [ape_opus[i] for i in idx]
            d_ref = [references[i] for i in idx]

            d_mt_s = score(d_mt, d_ref)
            d_ape_s = score(d_ape, d_ref)
            d = d_ape_s['bleu'] - d_mt_s['bleu']
            print(f"{domain:<20} {d_mt_s['bleu']:>15.2f} {d_ape_s['bleu']:>15.2f} {d:>+10.2f}")

    # Sample corrections
    print(f"\n{'=' * 70}")
    print("SAMPLE CORRECTIONS")
    print(f"{'=' * 70}")

    # Find examples where APE changed the output meaningfully
    changed = [(i, r) for i, r in enumerate(test_data) if ape_opus[i] != mt_opus[i]]
    print(f"\nAPE changed {len(changed):,}/{len(test_data):,} sentences ({100*len(changed)/len(test_data):.0f}%)")

    for idx, row in changed[:10]:
        print(f"\n--- Example (domain: {row['domain']}) ---")
        print(f"  DE:      {row['src'][:120]}")
        print(f"  opus-mt: {mt_opus[idx][:120]}")
        print(f"  APE:     {ape_opus[idx][:120]}")
        print(f"  Human:   {row['ref'][:120]}")

    # Save results
    if args.save_results:
        results = {
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "test_samples": len(test_data),
            "opus_mt": opus_scores,
            "ape_opus": ape_opus_scores,
            "deepl": deepl_scores,
            "ape_deepl": ape_deepl_scores,
        }
        results_path = os.path.join(os.path.dirname(args.model), "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    print(f"\nEvaluation complete.")


if __name__ == "__main__":
    main()
