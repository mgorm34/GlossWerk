"""
GlossWerk - Patent Terminology Extractor
Mines consistent DE→EN term mappings from your parallel training data.

Method:
  1. Word-align German source and English reference using statistical co-occurrence
  2. Find German words that consistently map to the same English word(s)
  3. Filter by frequency and consistency thresholds
  4. Output a curated terminology file

This gives you empirical patent terminology — terms that actually appear
in your data with verified translations, not generic dictionary entries.

Usage:
    python 09_extract_terminology.py
    python 09_extract_terminology.py --min_freq 20 --min_consistency 0.7
    python 09_extract_terminology.py --output "C:\glosswerk\data\patent_terms.tsv"
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

# Common function words to skip (DE and EN)
DE_STOPWORDS = {
    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'einem',
    'einen', 'eines', 'und', 'oder', 'aber', 'ist', 'sind', 'wird', 'werden',
    'wurde', 'wurden', 'hat', 'haben', 'hatte', 'kann', 'können', 'soll',
    'sollen', 'muss', 'müssen', 'mit', 'von', 'für', 'auf', 'bei', 'nach',
    'aus', 'über', 'unter', 'durch', 'als', 'wie', 'wenn', 'dass', 'nicht',
    'auch', 'noch', 'nur', 'mehr', 'sehr', 'schon', 'so', 'da', 'hier',
    'dort', 'wo', 'was', 'wer', 'welche', 'welcher', 'welches', 'diese',
    'dieser', 'dieses', 'jede', 'jeder', 'jedes', 'alle', 'keine', 'kein',
    'keiner', 'sich', 'es', 'er', 'sie', 'wir', 'ihr', 'ich', 'du',
    'man', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'ihre', 'ihrem',
    'ihren', 'ihrer', 'zum', 'zur', 'im', 'am', 'vom', 'beim', 'ins',
    'ans', 'aufs', 'ums', 'wobei', 'dabei', 'dazu', 'davon', 'dafür',
    'damit', 'daher', 'darum', 'darauf', 'darin', 'daraus', 'hierzu',
    'hierbei', 'bzw', 'sowie', 'bzw.', 'z.b.', 'z.', 'b.', 'd.h.',
    'vgl.', 'ca.', 'etc.', 'usw.', 'ggf.', 'u.a.', 'o.g.', 'sog.',
    'gem.', 'entsprechend', 'insbesondere', 'beispielsweise', 'vorzugsweise',
    'mindestens', 'wenigstens', 'zumindest', 'jeweils', 'beziehungsweise',
}

EN_STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'at', 'by', 'with',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'must', 'and', 'or', 'but',
    'not', 'no', 'nor', 'if', 'then', 'than', 'that', 'this', 'these',
    'those', 'which', 'who', 'whom', 'what', 'where', 'when', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'also', 'very', 'just', 'so', 'it', 'its',
    'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'our', 'your',
    'his', 'her', 'their', 'said', 'one', 'two', 'first', 'second',
}

# Known patent convention mappings (hand-curated)
PATENT_CONVENTIONS = [
    ("Fig.", "FIG.", "patent_convention"),
    ("Figur", "FIG.", "patent_convention"),
    ("Anspruch", "claim", "patent_convention"),
    ("Ansprüche", "claims", "patent_convention"),
    ("Ausführungsform", "embodiment", "patent_convention"),
    ("Ausführungsbeispiel", "exemplary embodiment", "patent_convention"),
    ("Ausführungsformen", "embodiments", "patent_convention"),
    ("erfindungsgemäß", "according to the invention", "patent_convention"),
    ("dadurch gekennzeichnet", "characterized in that", "patent_convention"),
    ("gekennzeichnet durch", "characterized by", "patent_convention"),
    ("umfasst", "comprises", "patent_convention"),
    ("umfassend", "comprising", "patent_convention"),
    ("aufweist", "comprises", "patent_convention"),
    ("aufweisend", "comprising", "patent_convention"),
    ("vorgesehen", "provided", "patent_convention"),
    ("Vorrichtung", "device", "patent_convention"),
    ("Verfahren", "method", "patent_convention"),
    ("Einrichtung", "apparatus", "patent_convention"),
    ("Anordnung", "arrangement", "patent_convention"),
    ("Zusammensetzung", "composition", "patent_convention"),
    ("Substrat", "substrate", "patent_convention"),
    ("Schicht", "layer", "patent_convention"),
    ("Beschichtung", "coating", "patent_convention"),
    ("Oberfläche", "surface", "patent_convention"),
    ("Querschnitt", "cross-section", "patent_convention"),
    ("Längsrichtung", "longitudinal direction", "patent_convention"),
    ("Querrichtung", "transverse direction", "patent_convention"),
    ("Umfangsrichtung", "circumferential direction", "patent_convention"),
    ("Wandstärke", "wall thickness", "patent_convention"),
    ("Drehzahl", "rotational speed", "patent_convention"),
    ("Drehmoment", "torque", "patent_convention"),
    ("Wirkungsgrad", "efficiency", "patent_convention"),
    ("Wellenlänge", "wavelength", "patent_convention"),
    ("Brennstoffzelle", "fuel cell", "patent_convention"),
    ("Halbleiter", "semiconductor", "patent_convention"),
    ("Leiterplatte", "circuit board", "patent_convention"),
    ("Wärmetauscher", "heat exchanger", "patent_convention"),
    ("Dichtung", "seal", "patent_convention"),
    ("Lager", "bearing", "patent_convention"),
    ("Getriebe", "transmission", "patent_convention"),
    ("Kolben", "piston", "patent_convention"),
    ("Ventil", "valve", "patent_convention"),
    ("Düse", "nozzle", "patent_convention"),
    ("Gehäuse", "housing", "patent_convention"),
    ("Flansch", "flange", "patent_convention"),
    ("Gewinde", "thread", "patent_convention"),
    ("Bolzen", "bolt", "patent_convention"),
    ("Achse", "axis", "patent_convention"),
    ("Welle", "shaft", "patent_convention"),
    ("Hebel", "lever", "patent_convention"),
    ("Feder", "spring", "patent_convention"),
    ("Nocken", "cam", "patent_convention"),
    ("Zahnrad", "gear", "patent_convention"),
    ("Kupplung", "clutch", "patent_convention"),
    ("Bremse", "brake", "patent_convention"),
    ("Zylinder", "cylinder", "patent_convention"),
    ("Rohr", "tube", "patent_convention"),
    ("Leitung", "conduit", "patent_convention"),
    ("Kanal", "channel", "patent_convention"),
    ("Öffnung", "opening", "patent_convention"),
    ("Bohrung", "bore", "patent_convention"),
    ("Aussparung", "recess", "patent_convention"),
    ("Nut", "groove", "patent_convention"),
    ("Steg", "web", "patent_convention"),
    ("Rippe", "rib", "patent_convention"),
    ("Absatz", "shoulder", "patent_convention"),
    ("Fase", "chamfer", "patent_convention"),
    ("Radius", "radius", "patent_convention"),
    ("Krümmung", "curvature", "patent_convention"),
    ("Neigung", "inclination", "patent_convention"),
    ("Steuereinheit", "control unit", "patent_convention"),
    ("Recheneinheit", "computing unit", "patent_convention"),
    ("Speicher", "memory", "patent_convention"),
    ("Sensor", "sensor", "patent_convention"),
    ("Aktor", "actuator", "patent_convention"),
    ("Stellglied", "actuating element", "patent_convention"),
    ("Antrieb", "drive", "patent_convention"),
    ("Elektrode", "electrode", "patent_convention"),
    ("Kathode", "cathode", "patent_convention"),
    ("Anode", "anode", "patent_convention"),
    ("Elektrolyt", "electrolyte", "patent_convention"),
    ("Werkstück", "workpiece", "patent_convention"),
    ("Werkzeug", "tool", "patent_convention"),
    ("Spritzguss", "injection molding", "patent_convention"),
    ("Extruder", "extruder", "patent_convention"),
    ("Schmelze", "melt", "patent_convention"),
    ("Legierung", "alloy", "patent_convention"),
    ("Kunststoff", "plastic", "patent_convention"),
    ("Elastomer", "elastomer", "patent_convention"),
    ("Verbundwerkstoff", "composite material", "patent_convention"),
    ("Zugfestigkeit", "tensile strength", "patent_convention"),
    ("Druckfestigkeit", "compressive strength", "patent_convention"),
    ("Härte", "hardness", "patent_convention"),
    ("Viskosität", "viscosity", "patent_convention"),
    ("Lösungsmittel", "solvent", "patent_convention"),
    ("Katalysator", "catalyst", "patent_convention"),
    ("Reagenz", "reagent", "patent_convention"),
    ("Additiv", "additive", "patent_convention"),
    ("Bindemittel", "binder", "patent_convention"),
    ("Füllstoff", "filler", "patent_convention"),
    ("Pigment", "pigment", "patent_convention"),
    ("Emulsion", "emulsion", "patent_convention"),
    ("Suspension", "suspension", "patent_convention"),
    ("Dispersion", "dispersion", "patent_convention"),
]


def tokenize(text):
    """Simple word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def extract_cooccurrence_terms(db_path, min_freq=10, min_consistency=0.6,
                                max_pairs=None, sample_size=500000):
    """
    Extract consistent DE→EN word mappings from parallel data.
    
    For each German content word, track which English content words
    appear in the same sentence pair. Words that consistently co-occur
    with the same English word are likely term translations.
    """
    conn = sqlite3.connect(db_path)

    query = "SELECT src, ref FROM sentence_pairs WHERE split = 'train'"
    if max_pairs:
        query += f" LIMIT {max_pairs}"
    elif sample_size:
        query += f" ORDER BY RANDOM() LIMIT {sample_size}"

    rows = conn.execute(query).fetchall()
    conn.close()

    print(f"  Processing {len(rows):,} sentence pairs...")

    # Track: for each DE word, count how often each EN word appears in the same pair
    de_en_cooccur = defaultdict(Counter)
    de_freq = Counter()

    for i, (src, ref) in enumerate(rows):
        de_words = set(tokenize(src)) - DE_STOPWORDS
        en_words = set(tokenize(ref)) - EN_STOPWORDS

        # Skip very short/long sentences (noisy)
        if len(de_words) < 3 or len(en_words) < 3:
            continue
        if len(de_words) > 40 or len(en_words) > 40:
            continue

        for de_word in de_words:
            if len(de_word) < 4:  # Skip very short words
                continue
            de_freq[de_word] += 1
            for en_word in en_words:
                if len(en_word) < 3:
                    continue
                de_en_cooccur[de_word][en_word] += 1

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,}/{len(rows):,}")

    print(f"  Unique DE words tracked: {len(de_en_cooccur):,}")

    # Extract high-confidence pairs
    term_pairs = []

    for de_word, en_counts in de_en_cooccur.items():
        freq = de_freq[de_word]
        if freq < min_freq:
            continue

        # Get the most common English co-occurrence
        top_en, top_count = en_counts.most_common(1)[0]
        consistency = top_count / freq

        if consistency >= min_consistency:
            # Additional filters
            if top_en in EN_STOPWORDS:
                continue
            # Skip if DE and EN are identical (likely proper nouns or numbers)
            if de_word == top_en:
                continue
            # Skip pure numbers
            if de_word.isdigit() or top_en.isdigit():
                continue

            term_pairs.append({
                'de_term': de_word,
                'en_term': top_en,
                'frequency': freq,
                'consistency': consistency,
                'domain': 'patent',
                'source': 'extracted',
            })

    # Sort by frequency * consistency (most reliable first)
    term_pairs.sort(key=lambda x: x['frequency'] * x['consistency'], reverse=True)

    print(f"  Extracted {len(term_pairs):,} high-confidence term pairs")
    return term_pairs


def save_to_tsv(term_pairs, filepath):
    """Save terms to TSV for review and import."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("de_term\ten_term\tfrequency\tconsistency\tdomain\n")
        for t in term_pairs:
            f.write(f"{t['de_term']}\t{t['en_term']}\t{t['frequency']}\t{t['consistency']:.2f}\t{t['domain']}\n")
    print(f"  Saved {len(term_pairs):,} terms to {filepath}")


def save_to_term_db(term_pairs, term_db_path):
    """Insert extracted terms into the terminology database."""
    conn = sqlite3.connect(term_db_path)
    cursor = conn.cursor()

    # Ensure table exists
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
    for t in term_pairs:
        de_lemma = t['de_term'].lower()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO terms (de_term, en_term, de_lemma, domain, source, reliability) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (t['de_term'], t['en_term'], de_lemma, t['domain'], 'extracted',
                 5 if t['consistency'] > 0.8 else 3)
            )
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    print(f"  Inserted {inserted:,} extracted terms into terminology DB")
    return inserted


def save_conventions_to_db(term_db_path):
    """Insert hand-curated patent conventions into the terminology database."""
    conn = sqlite3.connect(term_db_path)
    cursor = conn.cursor()

    inserted = 0
    for de_term, en_term, domain in PATENT_CONVENTIONS:
        de_lemma = de_term.lower()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO terms (de_term, en_term, de_lemma, domain, source, reliability) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (de_term, en_term, de_lemma, domain, 'curated', 5)
            )
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    print(f"  Inserted {inserted:,} curated patent conventions")
    return inserted


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    default_db = os.path.join(data_dir, "glosswerk_patent.db")
    default_term_db = os.path.join(data_dir, "glosswerk_terminology.db")
    default_output = os.path.join(data_dir, "extracted_patent_terms.tsv")

    parser = argparse.ArgumentParser(description="Extract patent terminology from training data")
    parser.add_argument("--db", type=str, default=default_db)
    parser.add_argument("--term_db", type=str, default=default_term_db)
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--min_freq", type=int, default=10,
                        help="Minimum frequency for a term pair (default: 10)")
    parser.add_argument("--min_consistency", type=float, default=0.6,
                        help="Minimum consistency ratio (default: 0.6)")
    parser.add_argument("--sample_size", type=int, default=500000,
                        help="Number of sentence pairs to sample (default: 500000)")
    parser.add_argument("--skip_conventions", action="store_true",
                        help="Skip inserting hand-curated patent conventions")
    args = parser.parse_args()

    print("=" * 60)
    print("GlossWerk - Patent Terminology Extractor")
    print("=" * 60)

    # Step 1: Insert hand-curated patent conventions
    if not args.skip_conventions:
        print(f"\n--- Inserting curated patent conventions ---")
        save_conventions_to_db(args.term_db)

    # Step 2: Extract terms from parallel data
    print(f"\n--- Extracting terms from training data ---")
    term_pairs = extract_cooccurrence_terms(
        args.db,
        min_freq=args.min_freq,
        min_consistency=args.min_consistency,
        sample_size=args.sample_size,
    )

    # Step 3: Save to TSV for manual review
    save_to_tsv(term_pairs, args.output)

    # Step 4: Insert into terminology database
    print(f"\n--- Adding to terminology database ---")
    save_to_term_db(term_pairs, args.term_db)

    # Show top terms
    print(f"\n--- Top 30 extracted patent terms ---")
    print(f"{'German':<30s} {'English':<25s} {'Freq':>6s} {'Consistency':>12s}")
    print("-" * 75)
    for t in term_pairs[:30]:
        print(f"{t['de_term']:<30s} {t['en_term']:<25s} {t['frequency']:>6d} {t['consistency']:>11.0%}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total extracted: {len(term_pairs):,} term pairs")
    print(f"TSV for review: {args.output}")
    print(f"Terms added to: {args.term_db}")
    print(f"\nReview the TSV file and remove any bad entries, then run:")
    print(f"  python 07_build_terminology.py --stats")
    print(f"  python 08_augment_training_data.py --domain patent")
    print("=" * 60)


if __name__ == "__main__":
    main()
