"""
Step 6: Progress tracker - log every training run and evaluation.
Maintains a CSV of all experiments for tracking improvement over time.

Usage:
    # Log a new result:
    python 06_track_progress.py --log --name "patent_stageA_v1" --bleu 25.3 --chrf 52.1 --notes "First patent run"

    # View all results:
    python 06_track_progress.py --show

    # Compare two runs:
    python 06_track_progress.py --compare "patent_stageA_v1" "patent_stageB_v1"
"""

import argparse
import csv
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER_PATH = os.path.join(PROJECT_ROOT, "experiments.csv")
FIELDS = ["timestamp", "name", "data", "model", "bleu_baseline", "bleu_ape", "bleu_delta",
          "chrf_baseline", "chrf_ape", "chrf_delta", "train_samples", "test_samples",
          "epochs", "batch_size", "training_hours", "notes"]


def init_tracker():
    if not os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
        print(f"Created tracker: {TRACKER_PATH}")


def log_result(args):
    init_tracker()
    row = {
        "timestamp": datetime.now().isoformat(),
        "name": args.name,
        "data": args.data or "",
        "model": args.model or "",
        "bleu_baseline": args.bleu_baseline or "",
        "bleu_ape": args.bleu or "",
        "bleu_delta": round(float(args.bleu) - float(args.bleu_baseline), 2) if args.bleu_baseline else "",
        "chrf_baseline": args.chrf_baseline or "",
        "chrf_ape": args.chrf or "",
        "chrf_delta": round(float(args.chrf) - float(args.chrf_baseline), 2) if args.chrf_baseline else "",
        "train_samples": args.train_samples or "",
        "test_samples": args.test_samples or "",
        "epochs": args.epochs or "",
        "batch_size": args.batch_size or "",
        "training_hours": args.hours or "",
        "notes": args.notes or "",
    }

    with open(TRACKER_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)

    print(f"Logged: {args.name} | BLEU: {args.bleu} | chrF: {args.chrf}")


def show_results():
    if not os.path.exists(TRACKER_PATH):
        print("No experiments logged yet.")
        return

    with open(TRACKER_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No experiments logged yet.")
        return

    print(f"\n{'=' * 90}")
    print(f"{'Name':<25} {'BLEU base':>10} {'BLEU APE':>10} {'Delta':>8} {'chrF APE':>10} {'Samples':>10} {'Notes'}")
    print("-" * 90)

    for r in rows:
        print(f"{r['name']:<25} {r['bleu_baseline']:>10} {r['bleu_ape']:>10} {r['bleu_delta']:>8} "
              f"{r['chrf_ape']:>10} {r['train_samples']:>10} {r['notes'][:30]}")

    print(f"\n{len(rows)} experiments logged.")
    print(f"Tracker: {TRACKER_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--bleu", type=float)
    parser.add_argument("--chrf", type=float)
    parser.add_argument("--bleu_baseline", type=float, default=None)
    parser.add_argument("--chrf_baseline", type=float, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()

    if args.show:
        show_results()
    elif args.log:
        if not args.name or args.bleu is None:
            print("--log requires --name and --bleu at minimum")
            return
        log_result(args)
    else:
        show_results()


if __name__ == "__main__":
    main()
