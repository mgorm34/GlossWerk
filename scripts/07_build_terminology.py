"""
GlossWerk - Terminology Database Builder
Downloads and processes terminology from multiple sources into a unified lookup DB.

Sources:
  - IATE (Inter-Active Terminology for Europe) - TBX format
  - WIPO Pearl - CSV export
  - Custom glossaries - TSV format

Usage:
    python 07_build_terminology.py --source iate
    python 07_build_terminology.py --source wipo
    python 07_build_terminology.py --source custom --file my_glossary.tsv
    python 07_build_terminology.py --stats
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TERM_DB_PATH = os.path.join(DATA_DIR, "glosswerk_terminology.db")

# IATE download info
IATE_DOWNLOAD_URL = "https://iate.europa.eu/em-api/downloads/public"
IATE_HELP = """
IATE cannot be downloaded automatically via script.
To download IATE:
  1. Go to https://iate.europa.eu/download-iate
  2. Select languages: German (de) and English (en)
  3. Select export format: TBX
  4. Download and save to: {data_dir}/raw/iate/
  5. Re-run this script with --source iate --file <path_to_tbx>
""".format(data_dir=DATA_DIR)

# Domain mapping for IATE subject fields
IATE_DOMAIN_MAP = {
    "SCIENCE": "patent",
    "INDUSTRY": "patent",
    "CHEMISTRY": "patent",
    "ELECTRONICS": "technical",
    "INFORMATION TECHNOLOGY": "technical",
    "MECHANICS": "patent",
    "MEDICAL": "medical",
    "HEALTH": "medical",
    "PHARMACEUTICAL": "medical",
    "LAW": "legal",
    "POLITICS": "legal",
    "EUROPEAN UNION": "legal",
    "FINANCE": "finance",
    "ECONOMICS": "finance",
    "TRADE": "finance",
}


def init_db():
    """Create terminology database."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(TERM_DB_PATH)
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

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_de_term ON terms(de_term)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_de_lemma ON terms(de_lemma)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_domain ON terms(domain)
    """)

    conn.commit()
    return conn


def clean_term(text):
    """Clean and normalize a term."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # Skip very short or very long terms
    if len(text) < 2 or len(text) > 200:
        return None
    # Skip terms that are mostly numbers or punctuation
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < len(text) * 0.4:
        return None
    return text


def simple_german_lemma(term):
    """
    Simple lemmatization for German terms.
    Handles common suffixes. Not perfect but catches most inflections
    without requiring spaCy as a dependency.
    """
    term_lower = term.lower()

    # Plural endings
    for suffix in ['en', 'er', 'es', 'e', 'n', 's']:
        if term_lower.endswith(suffix) and len(term_lower) > len(suffix) + 3:
            candidate = term_lower[:-len(suffix)]
            # Don't strip if it leaves a very short word
            if len(candidate) >= 3:
                return candidate

    return term_lower


def parse_iate_tbx(filepath):
    """Parse IATE TBX (TermBase eXchange) format."""
    print(f"  Parsing IATE TBX: {filepath}")

    terms = []
    skipped = Counter()

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        print("  Trying to parse as large file...")
        # For very large files, use iterparse
        terms = parse_iate_tbx_iterative(filepath)
        return terms

    # Find namespace
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    # Iterate through term entries
    for entry in root.iter(f'{ns}termEntry'):
        de_terms = []
        en_terms = []
        domains = []

        # Get subject field (domain)
        for descrip in entry.iter(f'{ns}descrip'):
            if descrip.get('type') == 'subjectField':
                subject = (descrip.text or '').upper()
                for key, domain in IATE_DOMAIN_MAP.items():
                    if key in subject:
                        domains.append(domain)
                        break

        # Get terms by language
        for langset in entry.iter(f'{ns}langSet'):
            lang = langset.get(f'{ns}lang', langset.get('lang', langset.get('{http://www.w3.org/XML/1998/namespace}lang', '')))
            for term_elem in langset.iter(f'{ns}term'):
                term_text = clean_term(term_elem.text)
                if term_text:
                    if 'de' in lang.lower():
                        de_terms.append(term_text)
                    elif 'en' in lang.lower():
                        en_terms.append(term_text)

        # Create all DE-EN pairs
        domain = domains[0] if domains else "general"
        for de in de_terms:
            for en in en_terms:
                terms.append({
                    'de_term': de,
                    'en_term': en,
                    'domain': domain,
                    'source': 'iate',
                })

    print(f"  Extracted {len(terms):,} term pairs")
    return terms


def parse_iate_tbx_iterative(filepath):
    """Parse large IATE TBX files using iterative parsing to save memory."""
    terms = []
    current_entry = {'de': [], 'en': [], 'domains': []}

    for event, elem in ET.iterparse(filepath, events=['start', 'end']):
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if event == 'start' and tag == 'termEntry':
            current_entry = {'de': [], 'en': [], 'domains': []}

        elif event == 'end' and tag == 'descrip':
            if elem.get('type') == 'subjectField':
                subject = (elem.text or '').upper()
                for key, domain in IATE_DOMAIN_MAP.items():
                    if key in subject:
                        current_entry['domains'].append(domain)
                        break

        elif event == 'end' and tag == 'langSet':
            lang = ''
            for attr in elem.attrib:
                if 'lang' in attr.lower():
                    lang = elem.get(attr, '')

            for term_elem in elem.iter():
                term_tag = term_elem.tag.split('}')[-1] if '}' in term_elem.tag else term_elem.tag
                if term_tag == 'term':
                    term_text = clean_term(term_elem.text)
                    if term_text:
                        if 'de' in lang.lower():
                            current_entry['de'].append(term_text)
                        elif 'en' in lang.lower():
                            current_entry['en'].append(term_text)

        elif event == 'end' and tag == 'termEntry':
            domain = current_entry['domains'][0] if current_entry['domains'] else 'general'
            for de in current_entry['de']:
                for en in current_entry['en']:
                    terms.append({
                        'de_term': de,
                        'en_term': en,
                        'domain': domain,
                        'source': 'iate',
                    })
            elem.clear()

    print(f"  Extracted {len(terms):,} term pairs (iterative)")
    return terms


def parse_wipo_csv(filepath):
    """Parse WIPO Pearl terminology export (CSV/TSV)."""
    print(f"  Parsing WIPO: {filepath}")
    terms = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # WIPO exports vary in column names - try common patterns
            de_term = None
            en_term = None

            for key, val in row.items():
                key_lower = key.lower()
                if 'german' in key_lower or key_lower == 'de':
                    de_term = clean_term(val)
                elif 'english' in key_lower or key_lower == 'en':
                    en_term = clean_term(val)

            if de_term and en_term:
                terms.append({
                    'de_term': de_term,
                    'en_term': en_term,
                    'domain': 'patent',
                    'source': 'wipo',
                })

    print(f"  Extracted {len(terms):,} term pairs")
    return terms


def parse_custom_tsv(filepath):
    """Parse a custom glossary in TSV format: de_term<TAB>en_term<TAB>domain (optional)."""
    print(f"  Parsing custom glossary: {filepath}")
    terms = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)  # Skip header if present

        for row in reader:
            if len(row) >= 2:
                de_term = clean_term(row[0])
                en_term = clean_term(row[1])
                domain = row[2].strip() if len(row) > 2 else 'custom'

                if de_term and en_term:
                    terms.append({
                        'de_term': de_term,
                        'en_term': en_term,
                        'domain': domain,
                        'source': 'custom',
                    })

    print(f"  Extracted {len(terms):,} term pairs")
    return terms


def insert_terms(conn, terms):
    """Insert terms into the database, skipping duplicates."""
    cursor = conn.cursor()
    inserted = 0
    skipped = 0

    for term in terms:
        de_lemma = simple_german_lemma(term['de_term'])
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO terms (de_term, en_term, de_lemma, domain, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (term['de_term'], term['en_term'], de_lemma, term['domain'], term['source'])
            )
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except sqlite3.Error:
            skipped += 1

    conn.commit()
    print(f"  Inserted: {inserted:,} | Skipped duplicates: {skipped:,}")
    return inserted


def show_stats(conn):
    """Show terminology database statistics."""
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
    print(f"\nTerminology Database: {TERM_DB_PATH}")
    print(f"Total term pairs: {total:,}")

    print(f"\nBy source:")
    for source, count in cursor.execute(
        "SELECT source, COUNT(*) FROM terms GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"  {source:<20s} {count:>10,}")

    print(f"\nBy domain:")
    for domain, count in cursor.execute(
        "SELECT domain, COUNT(*) FROM terms GROUP BY domain ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"  {domain:<20s} {count:>10,}")

    print(f"\nSample terms:")
    for row in cursor.execute(
        "SELECT de_term, en_term, domain, source FROM terms ORDER BY RANDOM() LIMIT 10"
    ).fetchall():
        print(f"  {row[0]:<30s} \u2192 {row[1]:<30s} [{row[2]}, {row[3]}]")


def main():
    parser = argparse.ArgumentParser(description="Build GlossWerk terminology database")
    parser.add_argument("--source", choices=["iate", "wipo", "custom"],
                        help="Terminology source to process")
    parser.add_argument("--file", type=str, help="Path to terminology file")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    args = parser.parse_args()

    conn = init_db()

    if args.stats:
        show_stats(conn)
        conn.close()
        return

    if not args.source:
        print("Specify --source (iate/wipo/custom) or --stats")
        conn.close()
        return

    print("=" * 60)
    print("GlossWerk - Terminology Database Builder")
    print("=" * 60)

    if args.source == "iate":
        if not args.file:
            print(IATE_HELP)
            conn.close()
            return
        terms = parse_iate_tbx(args.file)

    elif args.source == "wipo":
        if not args.file:
            print("WIPO Pearl download:")
            print("  1. Go to https://www.wipo.int/reference/en/wipopearl/")
            print("  2. Search for terms in your domain")
            print("  3. Export as CSV")
            print("  4. Re-run with --source wipo --file <path>")
            conn.close()
            return
        terms = parse_wipo_csv(args.file)

    elif args.source == "custom":
        if not args.file:
            print("Provide a TSV file: de_term<TAB>en_term<TAB>domain")
            conn.close()
            return
        terms = parse_custom_tsv(args.file)

    if terms:
        insert_terms(conn, terms)
        show_stats(conn)

    conn.close()
    print("\nNext step: python 08_augment_training_data.py")


if __name__ == "__main__":
    main()
