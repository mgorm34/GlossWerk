"""
GlossWerk - Personal HTER Evaluation Sample Generator
Generates a spreadsheet with side-by-side comparisons for manual quality assessment.

Columns: German Source | DeepL Output | GlossWerk Output | Human Reference | Your Rating | Notes

Usage:
    python generate_hter_sample.py --model "C:\glosswerk\models\patent_ape_stageA\final" --sample_size 100
"""

import argparse
import os
import random
import sqlite3
import csv

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def generate_corrections(model, tokenizer, texts, device, batch_size=24, max_length=256):
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
            print(f"  Generated {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    return corrections


def main():
    parser = argparse.ArgumentParser(description="Generate HTER evaluation spreadsheet")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--db", type=str, default=r"C:\glosswerk\data\glosswerk_patent.db")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of sentences to sample")
    parser.add_argument("--output", type=str, default=r"C:\glosswerk\hter_evaluation.tsv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GlossWerk - HTER Evaluation Sample Generator")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    # Load test sentences that have DeepL translations
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT src, mt_deepl, ref FROM sentence_pairs "
        "WHERE split = 'test' AND mt_deepl IS NOT NULL"
    ).fetchall()
    conn.close()

    print(f"Test sentences with DeepL: {len(rows):,}")

    # Sample
    random.seed(args.seed)
    sample = random.sample(rows, min(args.sample_size, len(rows)))
    print(f"Sampled: {len(sample)} sentences")

    sources = [r[0] for r in sample]
    deepl_outputs = [r[1] for r in sample]
    references = [r[2] for r in sample]

    # Generate GlossWerk corrections on DeepL output
    print("\nGenerating GlossWerk corrections on DeepL output...")
    glosswerk_outputs = generate_corrections(model, tokenizer, deepl_outputs, device, args.batch_size)

    # Count how many were actually changed
    changed = sum(1 for d, g in zip(deepl_outputs, glosswerk_outputs) if d.strip() != g.strip())
    print(f"GlossWerk changed {changed}/{len(sample)} sentences ({100*changed/len(sample):.0f}%)")

    # Write TSV
    print(f"\nWriting to: {args.output}")
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "ID",
            "German_Source",
            "DeepL_Output",
            "GlossWerk_Output",
            "Human_Reference",
            "Changed",
            "Preferred (DeepL/GlossWerk/Equal)",
            "DeepL_Edits_Needed (0-5)",
            "GlossWerk_Edits_Needed (0-5)",
            "Notes"
        ])

        for i, (src, deepl, glosswerk, ref) in enumerate(
            zip(sources, deepl_outputs, glosswerk_outputs, references), 1
        ):
            was_changed = "Yes" if deepl.strip() != glosswerk.strip() else "No"
            writer.writerow([
                i,
                src,
                deepl,
                glosswerk,
                ref,
                was_changed,
                "",  # Preferred - you fill this in
                "",  # DeepL edits needed - you fill this in
                "",  # GlossWerk edits needed - you fill this in
                "",  # Notes - you fill this in
            ])

    print(f"\nDone! Open {args.output} in Excel.")
    print(f"\nScoring guide:")
    print(f"  Preferred: Which output is closer to publication-ready? (DeepL / GlossWerk / Equal)")
    print(f"  Edits Needed (0-5):")
    print(f"    0 = Perfect, no edits needed")
    print(f"    1 = Minor punctuation or spacing fix")
    print(f"    2 = One word or small phrase needs changing")
    print(f"    3 = Several words or a clause needs rework")
    print(f"    4 = Significant restructuring needed")
    print(f"    5 = Completely wrong, needs full retranslation")
    print(f"\nAfter scoring, compute:")
    print(f"  - % where GlossWerk preferred over DeepL")
    print(f"  - Average edits needed: DeepL vs GlossWerk")
    print(f"  - % where GlossWerk made things worse")


if __name__ == "__main__":
    main()
