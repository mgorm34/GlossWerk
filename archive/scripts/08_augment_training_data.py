"""
GlossWerk - Training Data Augmentation with Terminology
Reads the training database and terminology DB, matches terms in German source
sentences, and creates augmented training pairs with terminology hints.

Input format:  postedit: [MT output]
Output format: postedit: [MT output] || terms: Rastaufnahme=latching receptacle; Fig.=FIG.

Includes terminology dropout (random omission) for robustness.

Usage:
    python 08_augment_training_data.py
    python 08_augment_training_data.py --domain patent --dropout 0.25
    python 08_augment_training_data.py --stats
"""

import argparse
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict


# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAINING_DB = os.path.join(DATA_DIR, "glosswerk_patent.db")
TERM_DB = os.path.join(DATA_DIR, "glosswerk_terminology.db")
OUTPUT_DB = os.path.join(DATA_DIR, "glosswerk_patent_augmented.db")


def load_terminology(term_db_path, domain=None):
    """
    Load terminology into lookup structures.
    Returns:
      - exact_lookup: {lowercase_de_term: [(de_term, en_term), ...]}
      - lemma_lookup: {lemma: [(de_term, en_term), ...]}
      - compound_parts: {part_lemma: [(de_term, en_term), ...]}  (for compound decomposition)
    """
    conn = sqlite3.connect(term_db_path)

    query = "SELECT de_term, en_term, de_lemma, domain FROM terms"
    params = []
    if domain:
        query += " WHERE domain = ? OR domain = 'general'"
        params = [domain]

    rows = conn.execute(query, params).fetchall()
    conn.close()

    exact_lookup = defaultdict(list)
    lemma_lookup = defaultdict(list)

    for de_term, en_term, de_lemma, dom in rows:
        pair = (de_term, en_term)
        exact_lookup[de_term.lower()].append(pair)
        if de_lemma:
            lemma_lookup[de_lemma].append(pair)

    print(f"  Loaded {len(exact_lookup):,} exact terms, {len(lemma_lookup):,} lemma entries")
    return exact_lookup, lemma_lookup


def simple_german_lemma(word):
    """Simple German lemmatization matching 07_build_terminology.py."""
    word_lower = word.lower()
    for suffix in ['en', 'er', 'es', 'e', 'n', 's']:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 3:
            candidate = word_lower[:-len(suffix)]
            if len(candidate) >= 3:
                return candidate
    return word_lower


def decompose_compound(word):
    """
    Simple German compound word decomposition.
    Tries splitting at various points and checks if parts are meaningful.
    Returns list of potential sub-words (minimum 4 chars each).
    """
    word_lower = word.lower()
    if len(word_lower) < 8:  # Too short to be a meaningful compound
        return []

    parts = []
    min_part = 4

    # Try splitting at each position
    for i in range(min_part, len(word_lower) - min_part + 1):
        left = word_lower[:i]
        right = word_lower[i:]

        # Handle linking elements (fugen-s, fugen-n, etc.)
        for fugen in ['', 's', 'n', 'en', 'er', 'es']:
            if right.startswith(fugen) and len(right) > len(fugen) + min_part:
                remainder = right[len(fugen):]
                if len(left) >= min_part and len(remainder) >= min_part:
                    parts.append(left)
                    parts.append(remainder)

    return list(set(parts))


def find_terms_in_sentence(source_text, exact_lookup, lemma_lookup, max_terms=5):
    """
    Find terminology matches in a German source sentence.
    Returns list of (de_term, en_term) pairs found.
    """
    matches = []
    seen_en = set()  # Avoid duplicate English terms

    source_lower = source_text.lower()

    # 1. Exact phrase matching (multi-word terms first, longest match wins)
    for de_term_lower, pairs in sorted(exact_lookup.items(), key=lambda x: -len(x[0])):
        if de_term_lower in source_lower:
            for de_term, en_term in pairs:
                if en_term.lower() not in seen_en:
                    matches.append((de_term, en_term))
                    seen_en.add(en_term.lower())
                    break  # One match per DE term
            if len(matches) >= max_terms:
                return matches

    # 2. Word-level lemma matching
    words = re.findall(r'\b\w+\b', source_text)
    for word in words:
        if len(matches) >= max_terms:
            break

        word_lemma = simple_german_lemma(word)

        if word_lemma in lemma_lookup:
            for de_term, en_term in lemma_lookup[word_lemma]:
                if en_term.lower() not in seen_en:
                    matches.append((de_term, en_term))
                    seen_en.add(en_term.lower())
                    break

    # 3. Compound decomposition for unmatched long words
    if len(matches) < max_terms:
        for word in words:
            if len(word) >= 8 and len(matches) < max_terms:
                parts = decompose_compound(word)
                for part in parts:
                    part_lemma = simple_german_lemma(part)
                    if part_lemma in lemma_lookup:
                        for de_term, en_term in lemma_lookup[part_lemma]:
                            if en_term.lower() not in seen_en:
                                matches.append((de_term, en_term))
                                seen_en.add(en_term.lower())
                                break
                        break  # One compound match per word

    return matches


def format_term_hints(term_pairs):
    """Format term pairs as the terminology hint string."""
    if not term_pairs:
        return ""
    hints = "; ".join(f"{de}={en}" for de, en in term_pairs)
    return f" || terms: {hints}"


def augment_database(training_db, term_db, output_db, domain="patent", dropout_rate=0.25, seed=42):
    """
    Read training DB, match terminology, write augmented DB.
    """
    random.seed(seed)

    print(f"\n--- Loading terminology ---")
    exact_lookup, lemma_lookup = load_terminology(term_db, domain)

    if not exact_lookup and not lemma_lookup:
        print("ERROR: No terminology loaded. Run 07_build_terminology.py first.")
        sys.exit(1)

    print(f"\n--- Reading training data ---")
    conn_in = sqlite3.connect(training_db)
    total = conn_in.execute("SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
    print(f"  Total pairs: {total:,}")

    # Create output database
    if os.path.exists(output_db):
        os.remove(output_db)
    conn_out = sqlite3.connect(output_db)
    cursor_out = conn_out.cursor()

    cursor_out.execute("""
        CREATE TABLE sentence_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src TEXT NOT NULL,
            ref TEXT NOT NULL,
            mt_opus TEXT NOT NULL,
            mt_opus_augmented TEXT NOT NULL,
            mt_deepl TEXT,
            mt_deepl_augmented TEXT,
            ter REAL NOT NULL,
            domain TEXT NOT NULL,
            corpus TEXT NOT NULL,
            split TEXT NOT NULL,
            num_terms_matched INTEGER DEFAULT 0,
            terms_used TEXT
        )
    """)

    print(f"\n--- Augmenting training data ---")
    print(f"  Terminology dropout rate: {dropout_rate:.0%}")

    stats = Counter()
    batch = []
    batch_size = 10000

    cursor_in = conn_in.execute(
        "SELECT src, ref, mt_opus, mt_deepl, ter, domain, corpus, split FROM sentence_pairs"
    )

    for i, row in enumerate(cursor_in):
        src, ref, mt_opus, mt_deepl, ter, dom, corpus, split = row

        # Find terms in source sentence
        term_pairs = find_terms_in_sentence(src, exact_lookup, lemma_lookup)
        num_terms = len(term_pairs)
        stats['total'] += 1

        if num_terms > 0:
            stats['has_terms'] += 1
            stats['total_terms'] += num_terms

            # Apply terminology dropout
            if random.random() < dropout_rate:
                # Drop all terms for this sentence (model learns to work without hints)
                term_hint = ""
                terms_json = json.dumps([]) if 'json' in dir() else "[]"
                stats['dropped'] += 1
            else:
                # Randomly drop some individual terms too (partial hints)
                if len(term_pairs) > 1 and random.random() < 0.3:
                    keep_count = random.randint(1, len(term_pairs))
                    term_pairs = random.sample(term_pairs, keep_count)
                    stats['partial_drop'] += 1

                term_hint = format_term_hints(term_pairs)
                terms_json = str([(de, en) for de, en in term_pairs])
        else:
            term_hint = ""
            terms_json = "[]"
            stats['no_terms'] += 1

        # Build augmented MT strings
        mt_opus_aug = mt_opus + term_hint if term_hint else mt_opus
        mt_deepl_aug = (mt_deepl + term_hint) if (mt_deepl and term_hint) else mt_deepl

        batch.append((
            src, ref, mt_opus, mt_opus_aug, mt_deepl, mt_deepl_aug,
            ter, dom, corpus, split, num_terms, terms_json
        ))

        if len(batch) >= batch_size:
            cursor_out.executemany(
                "INSERT INTO sentence_pairs "
                "(src, ref, mt_opus, mt_opus_augmented, mt_deepl, mt_deepl_augmented, "
                "ter, domain, corpus, split, num_terms_matched, terms_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch
            )
            conn_out.commit()
            batch = []

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1:,}/{total:,}")

    # Final batch
    if batch:
        cursor_out.executemany(
            "INSERT INTO sentence_pairs "
            "(src, ref, mt_opus, mt_opus_augmented, mt_deepl, mt_deepl_augmented, "
            "ter, domain, corpus, split, num_terms_matched, terms_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch
        )
        conn_out.commit()

    conn_in.close()

    # Print stats
    print(f"\n{'=' * 60}")
    print(f"Augmentation Complete")
    print(f"{'=' * 60}")
    print(f"  Total sentences:          {stats['total']:,}")
    print(f"  With term matches:        {stats['has_terms']:,} ({100*stats['has_terms']/max(1,stats['total']):.1f}%)")
    print(f"  Without term matches:     {stats['no_terms']:,} ({100*stats['no_terms']/max(1,stats['total']):.1f}%)")
    print(f"  Total terms matched:      {stats['total_terms']:,}")
    print(f"  Avg terms per match:      {stats['total_terms']/max(1,stats['has_terms']):.1f}")
    print(f"  Full dropout applied:     {stats['dropped']:,}")
    print(f"  Partial dropout applied:  {stats['partial_drop']:,}")

    # Show samples
    print(f"\n--- Sample augmented pairs ---")
    samples = conn_out.execute(
        "SELECT mt_opus, mt_opus_augmented, ref, num_terms_matched "
        "FROM sentence_pairs WHERE num_terms_matched > 0 AND split = 'train' "
        "ORDER BY RANDOM() LIMIT 5"
    ).fetchall()

    for mt, mt_aug, ref, n_terms in samples:
        print(f"\n  Original:   {mt[:100]}")
        print(f"  Augmented:  {mt_aug[:150]}")
        print(f"  Reference:  {ref[:100]}")
        print(f"  Terms: {n_terms}")

    conn_out.close()
    print(f"\nOutput: {output_db}")
    print(f"Next step: python 03_train_patent_model.py --stage A --db \"{output_db}\"")


def show_stats(output_db):
    """Show augmentation statistics."""
    if not os.path.exists(output_db):
        print(f"No augmented database found at {output_db}")
        print("Run augmentation first.")
        return

    conn = sqlite3.connect(output_db)

    total = conn.execute("SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
    with_terms = conn.execute("SELECT COUNT(*) FROM sentence_pairs WHERE num_terms_matched > 0").fetchone()[0]

    print(f"\nAugmented Database: {output_db}")
    print(f"  Total pairs: {total:,}")
    print(f"  With terminology: {with_terms:,} ({100*with_terms/max(1,total):.1f}%)")

    print(f"\nTerm match distribution:")
    for n, count in conn.execute(
        "SELECT num_terms_matched, COUNT(*) FROM sentence_pairs GROUP BY num_terms_matched ORDER BY num_terms_matched"
    ).fetchall():
        bar = "#" * min(50, count // max(1, total // 200))
        print(f"  {n} terms: {count:>8,} {bar}")

    print(f"\nBy split:")
    for split, count, avg in conn.execute(
        "SELECT split, COUNT(*), AVG(num_terms_matched) FROM sentence_pairs GROUP BY split"
    ).fetchall():
        print(f"  {split:<6s}: {count:>10,} pairs, avg {avg:.2f} terms/sentence")

    conn.close()


def main():
    # Need json for terms storage
    import json

    parser = argparse.ArgumentParser(description="Augment training data with terminology")
    parser.add_argument("--training_db", type=str, default=TRAINING_DB)
    parser.add_argument("--term_db", type=str, default=TERM_DB)
    parser.add_argument("--output_db", type=str, default=OUTPUT_DB)
    parser.add_argument("--domain", type=str, default="patent")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Fraction of sentences where terminology is dropped (default: 0.25)")
    parser.add_argument("--stats", action="store_true", help="Show augmentation statistics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stats:
        show_stats(args.output_db)
        return

    print("=" * 60)
    print("GlossWerk - Training Data Terminology Augmentation")
    print("=" * 60)

    augment_database(
        args.training_db, args.term_db, args.output_db,
        domain=args.domain, dropout_rate=args.dropout, seed=args.seed
    )


if __name__ == "__main__":
    main()
