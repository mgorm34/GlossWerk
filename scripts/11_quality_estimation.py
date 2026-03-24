"""
GlossWerk - Quality Estimation with CometKiwi
Scores each segment after APE correction using CometKiwi (reference-free QE).
Triages segments into confidence buckets: publish / review / full edit.

CometKiwi scores each (source, translation) pair from 0.0 to 1.0.
Higher = better quality. No reference translation needed.

Setup:
    pip install unbabel-comet --break-system-packages
    huggingface-cli login
    # Accept license at: https://huggingface.co/Unbabel/wmt23-cometkiwi-da

Usage:
    # Score APE output on your test set
    python 11_quality_estimation.py --model "C:\glosswerk\models\patent_ape_stageA\final"

    # Score with custom thresholds
    python 11_quality_estimation.py --model "..." --threshold_high 0.85 --threshold_low 0.70

    # Score a single file (for the document pipeline)
    python 11_quality_estimation.py --score_file "C:\glosswerk\output.tsv"
"""

import argparse
import csv
import os
import sqlite3
import sys
import time

import torch
from comet import download_model, load_from_checkpoint
from transformers import T5ForConditionalGeneration, T5TokenizerFast


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_cometkiwi():
    """Download and load CometKiwi model."""
    print("Loading CometKiwi QE model...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    print("CometKiwi loaded.")
    return model


def score_translations(qe_model, sources, translations, batch_size=32):
    """
    Score translations using CometKiwi.
    Returns list of scores (0.0 to 1.0, higher = better).
    """
    data = [{"src": src, "mt": mt} for src, mt in zip(sources, translations)]

    print(f"  Scoring {len(data):,} segments with CometKiwi...")
    start = time.time()
    output = qe_model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(data)/elapsed:.0f} segments/sec)")

    return output.scores


def triage_segments(scores, threshold_high=0.82, threshold_low=0.68):
    """
    Assign each segment to a confidence bucket.
    Returns list of labels: 'publish', 'review', 'full_edit'
    """
    labels = []
    for score in scores:
        if score >= threshold_high:
            labels.append("publish")
        elif score >= threshold_low:
            labels.append("review")
        else:
            labels.append("full_edit")
    return labels


def generate_ape_corrections(model, tokenizer, texts, device, batch_size=24, max_length=256):
    """Run APE model on MT output."""
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

        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"  APE: {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    return corrections


def run_full_pipeline(args):
    """Run APE + QE on test data and output triaged results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("GlossWerk - APE + Quality Estimation Pipeline")
    print("=" * 60)

    # Load APE model
    print(f"\nLoading APE model: {args.model}")
    tokenizer = T5TokenizerFast.from_pretrained(args.model)
    ape_model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    ape_model.eval()

    # Load test data with DeepL
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT src, mt_deepl, ref FROM sentence_pairs "
        "WHERE split = 'test' AND mt_deepl IS NOT NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (args.limit,)
    ).fetchall()
    conn.close()

    sources = [r[0] for r in rows]
    deepl_outputs = [r[1] for r in rows]
    references = [r[2] for r in rows]

    print(f"Test segments: {len(rows):,}")

    # Step 1: Run APE corrections
    print(f"\n--- Step 1: APE Correction ---")
    ape_outputs = generate_ape_corrections(
        ape_model, tokenizer, deepl_outputs, device, args.batch_size
    )

    # Free APE model GPU memory before loading QE model
    del ape_model
    torch.cuda.empty_cache()

    # Step 2: Score with CometKiwi
    print(f"\n--- Step 2: Quality Estimation ---")
    qe_model = load_cometkiwi()

    # Score both DeepL and APE output
    print("\nScoring DeepL output:")
    deepl_scores = score_translations(qe_model, sources, deepl_outputs, args.qe_batch_size)

    print("\nScoring GlossWerk (APE) output:")
    ape_scores = score_translations(qe_model, sources, ape_outputs, args.qe_batch_size)

    # Step 3: Triage
    print(f"\n--- Step 3: Triage ---")
    ape_labels = triage_segments(ape_scores, args.threshold_high, args.threshold_low)
    deepl_labels = triage_segments(deepl_scores, args.threshold_high, args.threshold_low)

    # Stats
    print(f"\nThresholds: publish >= {args.threshold_high}, review >= {args.threshold_low}")

    print(f"\nDeepL triage:")
    for label in ["publish", "review", "full_edit"]:
        count = deepl_labels.count(label)
        print(f"  {label:<12s}: {count:>6,} ({100*count/len(deepl_labels):.1f}%)")

    print(f"\nGlossWerk triage:")
    for label in ["publish", "review", "full_edit"]:
        count = ape_labels.count(label)
        print(f"  {label:<12s}: {count:>6,} ({100*count/len(ape_labels):.1f}%)")

    # Score comparison
    avg_deepl = sum(deepl_scores) / len(deepl_scores)
    avg_ape = sum(ape_scores) / len(ape_scores)
    improved = sum(1 for d, a in zip(deepl_scores, ape_scores) if a > d)
    same = sum(1 for d, a in zip(deepl_scores, ape_scores) if abs(a - d) < 0.01)
    worse = sum(1 for d, a in zip(deepl_scores, ape_scores) if a < d - 0.01)

    print(f"\nQE Score Comparison:")
    print(f"  Avg DeepL QE:     {avg_deepl:.4f}")
    print(f"  Avg GlossWerk QE: {avg_ape:.4f}")
    print(f"  Improvement:      {avg_ape - avg_deepl:+.4f}")
    print(f"  Segments improved: {improved:,} ({100*improved/len(rows):.1f}%)")
    print(f"  Segments same:     {same:,} ({100*same/len(rows):.1f}%)")
    print(f"  Segments worse:    {worse:,} ({100*worse/len(rows):.1f}%)")

    # Segments that moved from non-publish to publish
    upgraded = sum(1 for d, a in zip(deepl_labels, ape_labels) if d != "publish" and a == "publish")
    downgraded = sum(1 for d, a in zip(deepl_labels, ape_labels) if d == "publish" and a != "publish")
    print(f"\n  Upgraded to publish:   {upgraded:,}")
    print(f"  Downgraded from publish: {downgraded:,}")
    print(f"  Net segments saved from human review: {upgraded - downgraded:,}")

    # Save detailed results
    output_path = os.path.join(DATA_DIR, "qe_pipeline_results.tsv")
    print(f"\nSaving detailed results to: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "German_Source", "DeepL_Output", "GlossWerk_Output", "Human_Reference",
            "DeepL_QE_Score", "GlossWerk_QE_Score", "QE_Delta",
            "DeepL_Triage", "GlossWerk_Triage", "Changed"
        ])
        for i in range(len(rows)):
            writer.writerow([
                sources[i], deepl_outputs[i], ape_outputs[i], references[i],
                f"{deepl_scores[i]:.4f}", f"{ape_scores[i]:.4f}",
                f"{ape_scores[i] - deepl_scores[i]:+.4f}",
                deepl_labels[i], ape_labels[i],
                "Yes" if deepl_outputs[i].strip() != ape_outputs[i].strip() else "No"
            ])

    # Show examples from each bucket
    print(f"\n{'=' * 60}")
    print("SAMPLE OUTPUTS BY TRIAGE BUCKET")
    print(f"{'=' * 60}")

    for label in ["publish", "review", "full_edit"]:
        examples = [(i, ape_scores[i]) for i in range(len(rows)) if ape_labels[i] == label]
        if examples:
            examples.sort(key=lambda x: -x[1])
            idx, score = examples[0]
            print(f"\n--- {label.upper()} (QE: {score:.3f}) ---")
            print(f"  DE:        {sources[idx][:100]}")
            print(f"  DeepL:     {deepl_outputs[idx][:100]}")
            print(f"  GlossWerk: {ape_outputs[idx][:100]}")
            print(f"  Reference: {references[idx][:100]}")

    print(f"\n{'=' * 60}")
    print(f"These numbers are your product demo.")
    print(f"The key metric: how many segments move from 'review/full_edit' to 'publish'")
    print(f"after GlossWerk correction. That's translator time saved.")
    print(f"{'=' * 60}")


def main():
    default_db = os.path.join(DATA_DIR, "glosswerk_patent.db")

    parser = argparse.ArgumentParser(description="GlossWerk QE Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to APE model")
    parser.add_argument("--db", type=str, default=default_db)
    parser.add_argument("--limit", type=int, default=1000,
                        help="Number of test segments to process (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=24, help="APE batch size")
    parser.add_argument("--qe_batch_size", type=int, default=32, help="CometKiwi batch size")
    parser.add_argument("--threshold_high", type=float, default=0.82,
                        help="QE score threshold for 'publish' bucket (default: 0.82)")
    parser.add_argument("--threshold_low", type=float, default=0.68,
                        help="QE score threshold for 'review' bucket (default: 0.68)")
    args = parser.parse_args()

    run_full_pipeline(args)


if __name__ == "__main__":
    main()
