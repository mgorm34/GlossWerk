"""
GlossWerk A/B Evaluation Script

Compares two prompt configurations by running translation + QE on the same
document and comparing the QE triage distributions.

Usage:
    python eval_ab.py --input patent.docx --api-key YOUR_KEY

    # Compare current prompts vs a modified version:
    python eval_ab.py --input patent.docx --api-key YOUR_KEY --label-a "current" --label-b "new-calque-rules"

    # Use a saved baseline instead of re-running A:
    python eval_ab.py --input patent.docx --api-key YOUR_KEY --baseline results_a.json

The script runs the full pipeline (translate → QE) twice with different configs
and prints a side-by-side comparison of green/orange/red distributions.

Before running variant B, edit the prompt_layers.py or translate.py prompt
and save. The script will pick up whatever is currently in the code.

Requires: anthropic, python-docx
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from translate import extract_text_from_docx, split_sentences, translate_document
from quality_estimate import evaluate_translations, compute_triage


def run_pipeline(sentences, api_key, model, glossary=None, label="run"):
    """Run translate + QE pipeline and return results."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    print(f"  Sentences: {len(sentences)}")

    # Translate
    print(f"\n  Translating...")
    translations = translate_document(
        sentences=sentences,
        api_key=api_key,
        model=model,
        glossary=glossary,
        batch_size=50,
        progress_callback=lambda c, t: print(f"    {c}/{t}"),
    )

    # QE
    print(f"\n  Running QE...")
    qe_results = evaluate_translations(
        translations=translations,
        api_key=api_key,
        model=model,
        batch_size=20,
        progress_callback=lambda c, t: print(f"    {c}/{t}"),
    )

    # Triage
    triage = compute_triage(qe_results)

    return {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "n_sentences": len(sentences),
        "translations": translations,
        "qe_results": qe_results,
        "triage": triage,
    }


def compare_results(result_a, result_b):
    """Print side-by-side comparison of two runs."""
    sa = result_a["triage"]["summary"]
    sb = result_b["triage"]["summary"]

    label_a = result_a["label"]
    label_b = result_b["label"]

    print(f"\n{'='*70}")
    print(f"  A/B COMPARISON")
    print(f"{'='*70}")
    print(f"  {'':30s}  {label_a:>15s}  {label_b:>15s}  {'Delta':>10s}")
    print(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*10}")

    for metric, key in [
        ("Green (publishable)", "green_pct"),
        ("Orange (quick review)", "orange_pct"),
        ("Red (full edit)", "red_pct"),
    ]:
        va = sa[key]
        vb = sb[key]
        delta = vb - va
        sign = "+" if delta > 0 else ""
        # For green, positive delta is good. For red, negative is good.
        print(f"  {metric:30s}  {va:14.1f}%  {vb:14.1f}%  {sign}{delta:9.1f}%")

    print()

    # Error type breakdown
    ea = sa.get("error_breakdown", {})
    eb = sb.get("error_breakdown", {})
    all_cats = sorted(set(list(ea.keys()) + list(eb.keys())))

    if all_cats:
        print(f"  Error breakdown (non-good segments):")
        print(f"  {'':30s}  {label_a:>15s}  {label_b:>15s}")
        print(f"  {'-'*30}  {'-'*15}  {'-'*15}")
        for cat in all_cats:
            ca = ea.get(cat, 0)
            cb = eb.get(cat, 0)
            print(f"  {cat:30s}  {ca:>15d}  {cb:>15d}")
        print()

    # Per-segment differences
    qa = {r["index"]: r for r in result_a["qe_results"]}
    qb = {r["index"]: r for r in result_b["qe_results"]}
    rating_order = {"good": 0, "minor": 1, "major": 2, "critical": 3}

    improved = []
    degraded = []
    for idx in sorted(qa.keys()):
        if idx not in qb:
            continue
        ra = rating_order.get(qa[idx]["rating"], 1)
        rb = rating_order.get(qb[idx]["rating"], 1)
        if rb < ra:
            improved.append((idx, qa[idx]["rating"], qb[idx]["rating"]))
        elif rb > ra:
            degraded.append((idx, qa[idx]["rating"], qb[idx]["rating"]))

    if improved:
        print(f"  IMPROVED segments ({len(improved)}):")
        for idx, old, new in improved[:10]:
            print(f"    Segment {idx}: {old} → {new}")
        if len(improved) > 10:
            print(f"    ... and {len(improved) - 10} more")
        print()

    if degraded:
        print(f"  DEGRADED segments ({len(degraded)}):")
        for idx, old, new in degraded[:10]:
            print(f"    Segment {idx}: {old} → {new}")
        if len(degraded) > 10:
            print(f"    ... and {len(degraded) - 10} more")
        print()

    net = len(improved) - len(degraded)
    print(f"  Net change: {'+' if net > 0 else ''}{net} segments improved")
    print(f"{'='*70}\n")

    return {
        "improved": len(improved),
        "degraded": len(degraded),
        "net": net,
        "green_delta": sb["green_pct"] - sa["green_pct"],
        "red_delta": sb["red_pct"] - sa["red_pct"],
    }


def main():
    parser = argparse.ArgumentParser(description="A/B eval for GlossWerk prompt changes")
    parser.add_argument("--input", required=True, help="German .docx file")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or ANTHROPIC_API_KEY env)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--glossary", default=None, help="Optional glossary TSV")
    parser.add_argument("--baseline", default=None, help="JSON file with saved baseline results (skip A run)")
    parser.add_argument("--label-a", default="baseline", help="Label for run A")
    parser.add_argument("--label-b", default="variant", help="Label for run B")
    parser.add_argument("--output", default=None, help="Save comparison results to JSON")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set ANTHROPIC_API_KEY")
        sys.exit(1)

    # Extract and split document
    from pathlib import Path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    raw_text = extract_text_from_docx(str(input_path))
    sentences = split_sentences(raw_text)
    print(f"Document: {len(sentences)} sentences, ~{len(raw_text.split())} words")

    # Load glossary if provided
    glossary = None
    if args.glossary:
        from translate import load_glossary_tsv
        glossary = load_glossary_tsv(args.glossary)
        print(f"Glossary: {len(glossary)} terms")

    # Run A (baseline)
    if args.baseline:
        print(f"\nLoading baseline from: {args.baseline}")
        with open(args.baseline, "r", encoding="utf-8") as f:
            result_a = json.load(f)
        result_a["label"] = args.label_a
    else:
        print(f"\n>>> Make sure {args.label_a} prompt is active, then press Enter...")
        input()
        result_a = run_pipeline(sentences, api_key, args.model, glossary, args.label_a)

        # Save baseline for future runs
        baseline_path = f"eval_{args.label_a}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(result_a, f, ensure_ascii=False, indent=2)
        print(f"\n  Baseline saved to: {baseline_path}")

    # Run B (variant)
    print(f"\n>>> Apply your prompt changes for '{args.label_b}', then press Enter...")
    input()
    result_b = run_pipeline(sentences, api_key, args.model, glossary, args.label_b)

    # Compare
    comparison = compare_results(result_a, result_b)

    # Save comparison
    if args.output:
        output = {
            "comparison": comparison,
            "result_a_summary": result_a["triage"]["summary"],
            "result_b_summary": result_b["triage"]["summary"],
            "label_a": args.label_a,
            "label_b": args.label_b,
            "document": str(input_path),
            "timestamp": datetime.now().isoformat(),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Comparison saved to: {args.output}")

    # Save variant results
    variant_path = f"eval_{args.label_b}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(variant_path, "w", encoding="utf-8") as f:
        json.dump(result_b, f, ensure_ascii=False, indent=2)
    print(f"Variant results saved to: {variant_path}")


if __name__ == "__main__":
    main()
