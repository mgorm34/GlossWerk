"""
GlossWerk - Error-Driven Terminology Extractor
Finds terms where opus-mt consistently makes the SAME wrong translation
and the human reference consistently provides the SAME correct translation.

These are the exact corrections the APE model needs terminology support for.

Method:
  1. For each sentence pair, find words in the human reference that are NOT
     in the opus-mt output (corrections the human made)
  2. Find words in the opus-mt output that are NOT in the human reference
     (mistakes the MT made)
  3. Track which MT mistakes consistently get replaced by which human corrections
  4. Cross-reference with German source to build DE → EN term pairs

Example:
  opus-mt says "rest picture" in 50 sentences
  human says "latching receptacle" in those same 50 sentences
  German source contains "Rastaufnahme" in those sentences
  → Extract: Rastaufnahme = latching receptacle (MT error: rest picture)

Usage:
    python 09_extract_corrections.py
    python 09_extract_corrections.py --min_freq 5 --min_consistency 0.6
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

EN_STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'at', 'by', 'with',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'must', 'and', 'or', 'but',
    'not', 'no', 'if', 'then', 'than', 'that', 'this', 'these', 'those',
    'which', 'who', 'whom', 'what', 'where', 'when', 'how', 'it', 'its',
    'he', 'she', 'they', 'we', 'you', 'also', 'so', 'very', 'just',
    'more', 'most', 'such', 'only', 'each', 'one', 'two', 'said',
}

DE_STOPWORDS = {
    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer',
    'einem', 'einen', 'eines', 'und', 'oder', 'aber', 'ist', 'sind',
    'wird', 'werden', 'wurde', 'hat', 'haben', 'kann', 'mit', 'von',
    'für', 'auf', 'bei', 'nach', 'aus', 'über', 'unter', 'durch', 'als',
    'wie', 'wenn', 'dass', 'nicht', 'auch', 'noch', 'nur', 'sich', 'es',
    'er', 'sie', 'wir', 'zum', 'zur', 'im', 'am', 'vom', 'ins', 'wobei',
    'dabei', 'sowie', 'bzw', 'daß',
}


def tokenize(text):
    """Simple word tokenization, preserving case for proper nouns."""
    return re.findall(r'\b\w+\b', text)


def tokenize_lower(text):
    """Tokenize and lowercase."""
    return re.findall(r'\b\w+\b', text.lower())


def extract_corrections(db_path, sample_size=500000, min_freq=5, min_consistency=0.5):
    """
    Find systematic corrections between opus-mt output and human reference.
    
    For each sentence pair:
      - mt_words = set of words in opus-mt output
      - ref_words = set of words in human reference
      - mt_only = words MT used that human didn't (MT's mistakes)
      - ref_only = words human used that MT didn't (human's corrections)
    
    Track which mt_only words consistently appear alongside which ref_only words.
    These are systematic error → correction patterns.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT src, mt_opus, ref FROM sentence_pairs "
        "WHERE split = 'train' ORDER BY RANDOM() LIMIT ?",
        (sample_size,)
    ).fetchall()
    conn.close()

    print(f"  Processing {len(rows):,} sentence pairs...")

    # Track: when MT uses word X and human uses word Y instead,
    # what German source words are present?
    # Structure: correction_patterns[(mt_word, ref_word)] = Counter of DE words
    correction_patterns = defaultdict(Counter)
    # How often does each (mt_word, ref_word) pair appear?
    correction_freq = Counter()
    # How often does each mt_word appear in mt-only position?
    mt_error_freq = Counter()

    for i, (src, mt, ref) in enumerate(rows):
        mt_words = set(tokenize_lower(mt))
        ref_words = set(tokenize_lower(ref))
        src_words = set(tokenize_lower(src))

        # Words MT got wrong (in MT but not in reference)
        mt_only = mt_words - ref_words - EN_STOPWORDS
        # Words human corrected to (in reference but not in MT)
        ref_only = ref_words - mt_words - EN_STOPWORDS

        # Filter out very short words and numbers
        mt_only = {w for w in mt_only if len(w) >= 3 and not w.isdigit()}
        ref_only = {w for w in ref_only if len(w) >= 3 and not w.isdigit()}

        # Skip if too many differences (likely misaligned or very different)
        if len(mt_only) > 15 or len(ref_only) > 15:
            continue

        # Track each MT error word
        for mt_word in mt_only:
            mt_error_freq[mt_word] += 1

        # Track co-occurring corrections
        # Key insight: in a single sentence, if MT says "picture" where human
        # says "receptacle", these are likely a correction pair
        de_content = src_words - DE_STOPWORDS
        de_content = {w for w in de_content if len(w) >= 4}

        for mt_word in mt_only:
            for ref_word in ref_only:
                correction_freq[(mt_word, ref_word)] += 1
                for de_word in de_content:
                    correction_patterns[(mt_word, ref_word)][de_word] += 1

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,}/{len(rows):,}")

    print(f"  Found {len(correction_freq):,} unique correction patterns")

    # Now extract high-confidence corrections
    # A good correction: (mt_word, ref_word) appears frequently AND
    # is associated with a consistent German source word
    results = []

    for (mt_word, ref_word), freq in correction_freq.items():
        if freq < min_freq:
            continue

        # What percentage of time this MT word appears does it get
        # corrected to this specific ref word?
        consistency = freq / max(1, mt_error_freq[mt_word])
        if consistency < min_consistency:
            continue

        # Find the most associated German source word
        de_counts = correction_patterns[(mt_word, ref_word)]
        if not de_counts:
            continue

        top_de, top_de_count = de_counts.most_common(1)[0]
        de_association = top_de_count / freq  # How consistently is this DE word present?

        # Skip if the German source word isn't consistently associated
        if de_association < 0.3:
            # Try to find the best DE word even with lower association
            # (corrections might span multiple DE source words)
            best_de = None
            best_score = 0
            for de_word, de_count in de_counts.most_common(5):
                score = de_count / freq
                if score > best_score and len(de_word) >= 5:
                    best_de = de_word
                    best_score = score
            if best_de and best_score >= 0.15:
                top_de = best_de
                de_association = best_score
            else:
                continue

        results.append({
            'de_term': top_de,
            'en_correct': ref_word,
            'en_mt_error': mt_word,
            'frequency': freq,
            'consistency': consistency,
            'de_association': de_association,
        })

    # Sort by frequency * consistency
    results.sort(key=lambda x: x['frequency'] * x['consistency'], reverse=True)

    # Deduplicate: if same DE word maps to multiple corrections, keep the strongest
    seen_de = {}
    deduplicated = []
    for r in results:
        de = r['de_term']
        if de not in seen_de:
            seen_de[de] = r
            deduplicated.append(r)
        else:
            # Keep the one with higher frequency
            if r['frequency'] > seen_de[de]['frequency']:
                deduplicated.remove(seen_de[de])
                seen_de[de] = r
                deduplicated.append(r)

    print(f"  High-confidence corrections: {len(deduplicated):,}")
    return deduplicated


def find_multiword_corrections(db_path, sample_size=200000):
    """
    Find multi-word correction patterns like "rest picture" → "latching receptacle".
    Uses bigram comparison between MT and reference.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT src, mt_opus, ref FROM sentence_pairs "
        "WHERE split = 'train' ORDER BY RANDOM() LIMIT ?",
        (sample_size,)
    ).fetchall()
    conn.close()

    print(f"\n  Searching for multi-word corrections in {len(rows):,} pairs...")

    def get_bigrams(text):
        words = tokenize_lower(text)
        return set(' '.join(words[i:i+2]) for i in range(len(words)-1))

    # Track bigram corrections
    bigram_corrections = defaultdict(Counter)
    bigram_mt_freq = Counter()

    for i, (src, mt, ref) in enumerate(rows):
        mt_bigrams = get_bigrams(mt)
        ref_bigrams = get_bigrams(ref)

        mt_only = mt_bigrams - ref_bigrams
        ref_only = ref_bigrams - mt_bigrams

        # Filter out bigrams with stopwords only
        mt_only = {b for b in mt_only if not all(w in EN_STOPWORDS for w in b.split())}
        ref_only = {b for b in ref_only if not all(w in EN_STOPWORDS for w in b.split())}

        if len(mt_only) > 10 or len(ref_only) > 10:
            continue

        for mt_bg in mt_only:
            bigram_mt_freq[mt_bg] += 1
            for ref_bg in ref_only:
                bigram_corrections[mt_bg][ref_bg] += 1

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,}/{len(rows):,}")

    # Extract consistent bigram corrections
    results = []
    for mt_bg, ref_counts in bigram_corrections.items():
        freq_total = bigram_mt_freq[mt_bg]
        if freq_total < 5:
            continue

        top_ref, top_count = ref_counts.most_common(1)[0]
        consistency = top_count / freq_total

        if consistency >= 0.4 and top_count >= 3:
            results.append({
                'mt_bigram': mt_bg,
                'ref_bigram': top_ref,
                'frequency': top_count,
                'consistency': consistency,
            })

    results.sort(key=lambda x: x['frequency'] * x['consistency'], reverse=True)
    print(f"  Multi-word corrections found: {len(results):,}")
    return results


def save_results(single_corrections, multi_corrections, output_dir):
    """Save all corrections to TSV files for review."""

    # Single-word corrections
    single_path = os.path.join(output_dir, "patent_term_corrections.tsv")
    with open(single_path, 'w', encoding='utf-8') as f:
        f.write("german_source\tcorrect_english\tmt_error\tfrequency\tconsistency\n")
        for r in single_corrections:
            f.write(f"{r['de_term']}\t{r['en_correct']}\t{r['en_mt_error']}\t"
                    f"{r['frequency']}\t{r['consistency']:.2f}\n")
    print(f"  Single-word corrections: {single_path}")

    # Multi-word corrections
    multi_path = os.path.join(output_dir, "patent_multiword_corrections.tsv")
    with open(multi_path, 'w', encoding='utf-8') as f:
        f.write("mt_error_phrase\tcorrect_phrase\tfrequency\tconsistency\n")
        for r in multi_corrections:
            f.write(f"{r['mt_bigram']}\t{r['ref_bigram']}\t"
                    f"{r['frequency']}\t{r['consistency']:.2f}\n")
    print(f"  Multi-word corrections: {multi_path}")

    return single_path, multi_path


def insert_into_term_db(corrections, term_db_path):
    """Insert validated corrections into terminology database."""
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
    for r in corrections:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO terms (de_term, en_term, de_lemma, domain, source, reliability) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (r['de_term'], r['en_correct'], r['de_term'].lower(), 'patent', 'error_extraction', 4)
            )
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    print(f"  Inserted {inserted:,} correction-based terms into terminology DB")
    return inserted


def main():
    default_db = os.path.join(DATA_DIR, "glosswerk_patent.db")
    default_term_db = os.path.join(DATA_DIR, "glosswerk_terminology.db")

    parser = argparse.ArgumentParser(description="Extract terminology from MT correction patterns")
    parser.add_argument("--db", type=str, default=default_db)
    parser.add_argument("--term_db", type=str, default=default_term_db)
    parser.add_argument("--sample_size", type=int, default=500000)
    parser.add_argument("--min_freq", type=int, default=5,
                        help="Minimum frequency for a correction pattern (default: 5)")
    parser.add_argument("--min_consistency", type=float, default=0.5,
                        help="Minimum consistency for a correction (default: 0.5)")
    parser.add_argument("--no_insert", action="store_true",
                        help="Don't insert into term DB, just generate TSV for review")
    args = parser.parse_args()

    print("=" * 60)
    print("GlossWerk - Error-Driven Terminology Extractor")
    print("=" * 60)
    print("Finding terms where MT consistently makes the same mistake")
    print("and human translators consistently provide the same fix.")

    # Single-word corrections
    print(f"\n--- Single-word correction patterns ---")
    single_corrections = extract_corrections(
        args.db, args.sample_size, args.min_freq, args.min_consistency
    )

    # Multi-word corrections
    print(f"\n--- Multi-word correction patterns ---")
    multi_corrections = find_multiword_corrections(args.db, min(args.sample_size, 200000))

    # Save to TSV for review
    print(f"\n--- Saving for review ---")
    single_path, multi_path = save_results(single_corrections, multi_corrections, DATA_DIR)

    # Show top corrections
    print(f"\n--- Top 30 single-word corrections ---")
    print(f"{'German':<25s} {'MT says':<20s} {'Should be':<20s} {'Freq':>5s} {'Cons':>5s}")
    print("-" * 80)
    for r in single_corrections[:30]:
        print(f"{r['de_term']:<25s} {r['en_mt_error']:<20s} {r['en_correct']:<20s} "
              f"{r['frequency']:>5d} {r['consistency']:>4.0%}")

    print(f"\n--- Top 20 multi-word corrections ---")
    print(f"{'MT phrase':<30s} {'Correct phrase':<30s} {'Freq':>5s} {'Cons':>5s}")
    print("-" * 70)
    for r in multi_corrections[:20]:
        print(f"{r['mt_bigram']:<30s} {r['ref_bigram']:<30s} "
              f"{r['frequency']:>5d} {r['consistency']:>4.0%}")

    # Insert into term DB (unless --no_insert)
    if not args.no_insert and single_corrections:
        print(f"\n--- Adding to terminology database ---")
        insert_into_term_db(single_corrections, args.term_db)

    print(f"\n{'=' * 60}")
    print(f"Review the TSV files, remove any bad entries, then run:")
    print(f"  python 07_build_terminology.py --stats")
    print(f"  python 08_augment_training_data.py --domain patent")
    print("=" * 60)


if __name__ == "__main__":
    main()
