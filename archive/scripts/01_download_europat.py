"""
Step 1: Download all domain corpora for GlossWerk APE.
Downloads patent, medical, legal, and general technical data.
Also downloads IATE terminology database (8M+ terms).

Usage:
    conda activate glosswerk
    cd C:\glosswerk\scripts
    python 01_download_europat.py

    # Download only specific corpora:
    python 01_download_europat.py --only europat emea

Downloads to: C:\glosswerk\data\raw\
"""

import argparse
import os
import gzip
import shutil
import urllib.request
import sys

DATA_DIR = r"C:\glosswerk\data\raw"
os.makedirs(DATA_DIR, exist_ok=True)

# All available corpora - OPUS download URLs (Moses format)
CORPORA = {
    # --- PATENTS ---
    "europat": {
        "src": "https://opus.nlpl.eu/download.php?f=EuroPat/v2/moses/de-en.txt.zip",
        "description": "European Patent translations (DE-EN)",
        "domain": "patent",
        "est_size": "large (millions of pairs)",
    },
    # --- MEDICAL / PHARMA ---
    "emea": {
        "src": "https://opus.nlpl.eu/download.php?f=EMEA/v3/moses/de-en.txt.zip",
        "description": "European Medicines Agency - drug/pharma (DE-EN)",
        "domain": "medical",
        "est_size": "medium (~1M pairs)",
    },
    # --- LEGAL ---
    "dgt": {
        "src": "https://opus.nlpl.eu/download.php?f=DGT/v2019/moses/de-en.txt.zip",
        "description": "EU Directorate-General for Translation - legal (DE-EN)",
        "domain": "legal",
        "est_size": "large (millions of pairs)",
    },
    "jrc_acquis": {
        "src": "https://opus.nlpl.eu/download.php?f=JRC-Acquis/v3.0/moses/de-en.txt.zip",
        "description": "EU Law / Acquis Communautaire (DE-EN)",
        "domain": "legal",
        "est_size": "large (~800k pairs)",
    },
    # --- TECHNICAL / ENGINEERING ---
    "kde4": {
        "src": "https://opus.nlpl.eu/download.php?f=KDE4/v2/moses/de-en.txt.zip",
        "description": "KDE software documentation - technical (DE-EN)",
        "domain": "technical",
        "est_size": "medium (~200k pairs)",
    },
    "gnome": {
        "src": "https://opus.nlpl.eu/download.php?f=GNOME/v1/moses/de-en.txt.zip",
        "description": "GNOME software documentation - technical (DE-EN)",
        "domain": "technical",
        "est_size": "small (~100k pairs)",
    },
    "ecb": {
        "src": "https://opus.nlpl.eu/download.php?f=ECB/v1/moses/de-en.txt.zip",
        "description": "European Central Bank - finance/economics (DE-EN)",
        "domain": "finance",
        "est_size": "small (~100k pairs)",
    },
}

# IATE terminology database (separate download - TBX format)
IATE_URL = "https://iate.europa.eu/em-api/downloads/public"
IATE_DIR = os.path.join(DATA_DIR, "iate")


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"  Downloading: {url}")
    print(f"  Saving to:   {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print("\n  Done!")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        print(f"\n  If automatic download fails, manually download from:")
        print(f"    {url}")
        print(f"  Save to: {dest_path}")
        return False


def extract_zip(zip_path, extract_dir):
    """Extract a zip file."""
    import zipfile
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print(f"  Extracted to: {extract_dir}")


def count_lines(filepath):
    """Count lines in a text file."""
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Download GlossWerk training corpora")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Download only specific corpora (e.g., --only europat emea)")
    parser.add_argument("--list", action="store_true", help="List available corpora and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("GlossWerk - Multi-Domain Corpus Downloader")
    print("=" * 60)

    if args.list:
        print(f"\n{'Name':<15} {'Domain':<12} {'Size':<25} {'Description'}")
        print("-" * 80)
        for name, info in CORPORA.items():
            print(f"{name:<15} {info['domain']:<12} {info['est_size']:<25} {info['description']}")
        return

    corpora_to_download = CORPORA
    if args.only:
        corpora_to_download = {k: v for k, v in CORPORA.items() if k in args.only}
        if not corpora_to_download:
            print(f"ERROR: None of {args.only} found. Use --list to see available corpora.")
            return

    total_corpora = len(corpora_to_download)
    successful = 0

    for i, (name, info) in enumerate(corpora_to_download.items(), 1):
        print(f"\n[{i}/{total_corpora}] --- {info['description']} ---")
        print(f"  Domain: {info['domain']} | Est. size: {info['est_size']}")
        corpus_dir = os.path.join(DATA_DIR, name)
        os.makedirs(corpus_dir, exist_ok=True)

        zip_path = os.path.join(corpus_dir, f"{name}_de-en.zip")

        # Download
        if not os.path.exists(zip_path):
            success = download_file(info["src"], zip_path)
            if not success:
                continue
        else:
            print(f"  Already downloaded: {zip_path}")

        # Extract
        extract_zip(zip_path, corpus_dir)

        # Count lines
        for f in os.listdir(corpus_dir):
            if f.endswith((".de", ".en")):
                fpath = os.path.join(corpus_dir, f)
                lines = count_lines(fpath)
                print(f"  {f}: {lines:,} lines")

        successful += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Downloads complete: {successful}/{total_corpora} corpora")
    print(f"Data saved to: {DATA_DIR}")
    print(f"\nDownloaded corpora by domain:")

    domains = {}
    for name, info in corpora_to_download.items():
        d = info["domain"]
        if d not in domains:
            domains[d] = []
        domains[d].append(name)

    for domain, names in sorted(domains.items()):
        print(f"  {domain}: {', '.join(names)}")

    print(f"\nNext step: python 02_build_patent_db.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
