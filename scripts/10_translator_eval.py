"""
GlossWerk - Translator Evaluation Workflow
Combined HTER evaluation + terminology extraction in one sitting.

Pulls sentences where the APE model made the biggest changes to DeepL output,
generates a spreadsheet where you evaluate AND extract terminology simultaneously.

Workflow:
  1. Run this script → generates evaluation spreadsheet
  2. Open in Excel, spend 2 hours working through rows
  3. For each sentence: score quality + capture any terminology corrections
  4. Run the companion import script to load your terms into the glossary

Output columns:
  A: ID
  B: German Source
  C: DeepL Output  
  D: GlossWerk Output
  E: Human Reference
  F: Changed (Yes/No)
  G: Preferred (DeepL / GlossWerk / Equal)
  H: GlossWerk Used Correct Term (Yes / No / N/A)
  I: Edits Needed for DeepL (0-5)
  J: Edits Needed for GlossWerk (0-5)
  K: German Term (if you spot a terminology issue)
  L: Wrong MT Term (what DeepL/MT said)
  M: Correct English Term (what it should be)
  N: Notes

Usage:
    python 10_translator_eval.py --model "C:\glosswerk\models\patent_ape_term_stageA\final"
    python 10_translator_eval.py --model "C:\glosswerk\models\patent_ape_stageA\final" --sample_size 200
    
    # After evaluation, import your terminology:
    python 10_translator_eval.py --import_terms "C:\glosswerk\hter_evaluation.tsv"
"""

import argparse
import csv
import os
import random
import sqlite3
import sys

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


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


def compute_edit_distance(a, b):
    """Simple word-level edit distance between two strings."""
    words_a = a.lower().split()
    words_b = b.lower().split()
    m, n = len(words_a), len(words_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words_a[i-1] == words_b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def generate_eval_spreadsheet(args):
    """Generate the evaluation spreadsheet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("GlossWerk - Translator Evaluation Workflow")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    # Load test sentences with DeepL translations
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT src, mt_deepl, mt_opus, ref FROM sentence_pairs "
        "WHERE split = 'test' AND mt_deepl IS NOT NULL"
    ).fetchall()
    conn.close()

    print(f"Test sentences with DeepL: {len(rows):,}")

    if not rows:
        print("ERROR: No test sentences with DeepL translations found.")
        print("Run 04_deepl_corrections.py first.")
        sys.exit(1)

    # Generate GlossWerk corrections on ALL DeepL test sentences
    print("\nGenerating GlossWerk corrections...")
    deepl_outputs = [r[1] for r in rows]
    glosswerk_outputs = generate_corrections(model, tokenizer, deepl_outputs, device, args.batch_size)

    # Compute edit distance for each pair (how much did GlossWerk change?)
    print("Computing edit distances...")
    scored = []
    for i, (row, gw) in enumerate(zip(rows, glosswerk_outputs)):
        src, deepl, opus, ref = row
        edit_dist = compute_edit_distance(deepl, gw)
        scored.append({
            'src': src,
            'deepl': deepl,
            'glosswerk': gw,
            'ref': ref,
            'edit_distance': edit_dist,
            'changed': deepl.strip() != gw.strip(),
        })

    # Strategy: sample across different edit distance ranges
    # We want a mix of: heavily changed, lightly changed, and unchanged
    changed = [s for s in scored if s['changed']]
    unchanged = [s for s in scored if not s['changed']]

    print(f"Changed: {len(changed):,} | Unchanged: {len(unchanged):,}")

    # Sort changed by edit distance (most changed first)
    changed.sort(key=lambda x: -x['edit_distance'])

    random.seed(args.seed)
    sample = []

    # 60% from high-edit-distance (most interesting corrections)
    high_edit = changed[:len(changed)//3]
    n_high = min(int(args.sample_size * 0.6), len(high_edit))
    sample.extend(random.sample(high_edit, n_high))

    # 25% from medium-edit-distance
    medium_edit = changed[len(changed)//3:2*len(changed)//3]
    n_med = min(int(args.sample_size * 0.25), len(medium_edit))
    if medium_edit:
        sample.extend(random.sample(medium_edit, n_med))

    # 15% from low-edit-distance or unchanged (sanity check)
    low_edit = changed[2*len(changed)//3:] + unchanged
    n_low = min(args.sample_size - len(sample), len(low_edit))
    if low_edit:
        sample.extend(random.sample(low_edit, n_low))

    random.shuffle(sample)  # Don't let the evaluator see the stratification

    print(f"Sampled {len(sample)} sentences for evaluation")

    # Write spreadsheet
    print(f"\nWriting to: {args.output}")
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "ID",
            "German_Source",
            "DeepL_Output",
            "GlossWerk_Output",
            "Human_Reference",
            "Changed",
            "Preferred (DeepL/GlossWerk/Equal)",
            "GW_Used_Correct_Term (Yes/No/NA)",
            "Edits_Needed_DeepL (0-5)",
            "Edits_Needed_GlossWerk (0-5)",
            "DE_Term (if terminology issue)",
            "Wrong_MT_Term",
            "Correct_EN_Term",
            "Notes",
        ])

        for i, s in enumerate(sample, 1):
            writer.writerow([
                i,
                s['src'],
                s['deepl'],
                s['glosswerk'],
                s['ref'],
                "Yes" if s['changed'] else "No",
                "",  # Preferred
                "",  # GW correct term
                "",  # Edits DeepL
                "",  # Edits GlossWerk
                "",  # DE term
                "",  # Wrong MT term
                "",  # Correct EN term
                "",  # Notes
            ])

    print(f"\n{'=' * 60}")
    print(f"EVALUATION INSTRUCTIONS")
    print(f"{'=' * 60}")
    print(f"""
Open {args.output} in Excel.

For each row, fill in:

  G: Preferred — Which is better? "DeepL" / "GlossWerk" / "Equal"
  
  H: GW Used Correct Term — If GlossWerk changed terminology,
     was the change correct? "Yes" / "No" / "NA" (if no term change)
  
  I: Edits Needed DeepL — How many edits to make DeepL output 
     publication-ready? (0=perfect, 5=rewrite)
  
  J: Edits Needed GlossWerk — Same scale for GlossWerk output
  
  K-M: TERMINOLOGY CAPTURE (the gold)
     If you spot a terminology issue in either output:
     K: The German term (e.g., Rastaufnahme)
     L: What MT said wrong (e.g., rest picture)  
     M: What it should be (e.g., latching receptacle)
     
     This builds your glossary as you evaluate!

  N: Notes — Anything else you notice

After finishing, run:
  python 10_translator_eval.py --import_terms "{args.output}"
  python 10_translator_eval.py --summarize "{args.output}"
""")


def import_terms(filepath, term_db_path):
    """Import terminology captured during evaluation into the glossary."""
    print("=" * 60)
    print("Importing terminology from evaluation")
    print("=" * 60)

    terms = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            de_term = (row.get('DE_Term (if terminology issue)', '') or '').strip()
            correct_en = (row.get('Correct_EN_Term', '') or '').strip()

            if de_term and correct_en:
                terms.append((de_term, correct_en))

    if not terms:
        print("No terminology entries found in the evaluation file.")
        print("Fill in columns K (DE_Term) and M (Correct_EN_Term) during evaluation.")
        return

    print(f"Found {len(terms)} terminology entries")

    conn = sqlite3.connect(term_db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            de_term TEXT NOT NULL,
            en_term TEXT NOT NULL,
            de_lemma TEXT,
            domain TEXT,
            source TEXT NOT NULL,
            reliability INTEGER DEFAULT 3,
            UNIQUE(de_term, en_term, source)
        )
    """)

    inserted = 0
    for de_term, en_term in terms:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO terms (de_term, en_term, de_lemma, domain, source, reliability) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (de_term, en_term, de_term.lower(), 'patent', 'translator_eval', 5)
            )
            if cursor.rowcount > 0:
                inserted += 1
                print(f"  + {de_term} = {en_term}")
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    print(f"\nInserted {inserted} new terms (reliability: 5/5 — translator verified)")
    print(f"Run: python 07_build_terminology.py --stats")


def summarize_eval(filepath):
    """Summarize evaluation results."""
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)

    total = len(rows)
    if total == 0:
        print("No data found.")
        return

    # Preference counts
    prefs = {'DeepL': 0, 'GlossWerk': 0, 'Equal': 0, '': 0}
    for r in rows:
        pref = (r.get('Preferred (DeepL/GlossWerk/Equal)', '') or '').strip()
        if pref in prefs:
            prefs[pref] += 1
        else:
            prefs[pref] = 1

    scored = total - prefs.get('', 0)

    print(f"\nTotal sentences: {total}")
    print(f"Scored: {scored}")

    if scored > 0:
        print(f"\nPreference:")
        print(f"  GlossWerk preferred: {prefs.get('GlossWerk', 0):>4d} ({100*prefs.get('GlossWerk',0)/scored:.0f}%)")
        print(f"  DeepL preferred:     {prefs.get('DeepL', 0):>4d} ({100*prefs.get('DeepL',0)/scored:.0f}%)")
        print(f"  Equal:               {prefs.get('Equal', 0):>4d} ({100*prefs.get('Equal',0)/scored:.0f}%)")

    # Edit scores
    deepl_edits = []
    gw_edits = []
    for r in rows:
        de = r.get('Edits_Needed_DeepL (0-5)', '').strip()
        ge = r.get('Edits_Needed_GlossWerk (0-5)', '').strip()
        if de and de.isdigit():
            deepl_edits.append(int(de))
        if ge and ge.isdigit():
            gw_edits.append(int(ge))

    if deepl_edits and gw_edits:
        avg_deepl = sum(deepl_edits) / len(deepl_edits)
        avg_gw = sum(gw_edits) / len(gw_edits)
        print(f"\nAverage edits needed:")
        print(f"  DeepL:     {avg_deepl:.2f}")
        print(f"  GlossWerk: {avg_gw:.2f}")
        print(f"  Reduction: {avg_deepl - avg_gw:.2f} fewer edits ({100*(avg_deepl-avg_gw)/max(0.01,avg_deepl):.0f}%)")

    # Terminology
    term_correct = 0
    term_wrong = 0
    term_na = 0
    for r in rows:
        tc = (r.get('GW_Used_Correct_Term (Yes/No/NA)', '') or '').strip().lower()
        if tc == 'yes':
            term_correct += 1
        elif tc == 'no':
            term_wrong += 1
        elif tc == 'na':
            term_na += 1

    if term_correct + term_wrong > 0:
        adherence = term_correct / (term_correct + term_wrong)
        print(f"\nTerminology adherence:")
        print(f"  Correct: {term_correct}")
        print(f"  Wrong:   {term_wrong}")
        print(f"  N/A:     {term_na}")
        print(f"  Adherence rate: {adherence:.0%}")

    # Terms captured
    terms_captured = sum(1 for r in rows if (r.get('DE_Term (if terminology issue)', '') or '').strip())
    print(f"\nTerminology entries captured: {terms_captured}")
    if terms_captured > 0:
        print(f"Run: python 10_translator_eval.py --import_terms \"{filepath}\"")

    # Changed vs unchanged
    changed = sum(1 for r in rows if r.get('Changed', '') == 'Yes')
    print(f"\nSentences GlossWerk changed: {changed}/{total} ({100*changed/total:.0f}%)")

    print(f"\n{'=' * 60}")
    print("These numbers are your product demo.")
    print("=" * 60)


def main():
    default_db = os.path.join(DATA_DIR, "glosswerk_patent.db")
    default_term_db = os.path.join(DATA_DIR, "glosswerk_terminology.db")
    default_output = os.path.join(PROJECT_ROOT, "hter_evaluation.tsv")

    parser = argparse.ArgumentParser(description="Translator evaluation workflow")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--db", type=str, default=default_db)
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--import_terms", type=str,
                        help="Import terminology from completed evaluation TSV")
    parser.add_argument("--summarize", type=str,
                        help="Summarize completed evaluation results")
    args = parser.parse_args()

    if args.import_terms:
        import_terms(args.import_terms, default_term_db)
    elif args.summarize:
        summarize_eval(args.summarize)
    elif args.model:
        generate_eval_spreadsheet(args)
    else:
        print("Provide --model to generate evaluation, --import_terms to import, or --summarize to view results")


if __name__ == "__main__":
    main()
