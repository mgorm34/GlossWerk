"""
GlossWerk - OPUS Corpus Downloader via API
Uses the OPUS API to find correct download URLs, then downloads and extracts.
Uses 'requests' library to avoid SSL certificate issues on Windows.
"""

import argparse
import json
import os
import zipfile
import requests

API_BASE = "https://opus.nlpl.eu/opusapi/"

CORPORA = {
    "europat":    {"name": "EuroPat",    "version": "latest", "domain": "patent"},
    "emea":       {"name": "EMEA",       "version": "latest", "domain": "medical"},
    "dgt":        {"name": "DGT",        "version": "latest", "domain": "legal"},
    "jrc_acquis": {"name": "JRC-Acquis", "version": "latest", "domain": "legal"},
    "kde4":       {"name": "KDE4",       "version": "latest", "domain": "technical"},
    "gnome":      {"name": "GNOME",      "version": "latest", "domain": "technical"},
    "ecb":        {"name": "ECB",        "version": "latest", "domain": "finance"},
}


def find_download_url(corpus_name):
    """Query OPUS API to find the moses download URL for de-en."""
    url = f"{API_BASE}?corpus={corpus_name}&source=de&target=en&preprocessing=moses"
    print(f"  Querying API: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "corpora" in data and len(data["corpora"]) > 0:
            best = data["corpora"][-1]
            dl_url = best.get("url", "")
            version = best.get("version", "unknown")
            size = best.get("size", "unknown")
            print(f"  Found: version={version}, size={size}")
            print(f"  URL: {dl_url}")
            return dl_url, version
        else:
            print(f"  No results from API. Response: {json.dumps(data, indent=2)[:500]}")
            return None, None
    except Exception as e:
        print(f"  API query failed: {e}")
        return None, None


def download_and_extract(url, corpus_key, out_dir):
    """Download zip and extract .de and .en files."""
    zip_path = os.path.join(out_dir, f"{corpus_key}.zip")

    print(f"  Downloading: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Downloaded: {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  Downloaded: {downloaded // (1024*1024)}MB", end="", flush=True)
        print()
    except Exception as e:
        print(f"  Download failed: {e}")
        return None, None

    # Extract
    print(f"  Extracting {zip_path}...")
    de_file = None
    en_file = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()
            print(f"  Archive contains: {names[:10]}{'...' if len(names) > 10 else ''}")

            for name in names:
                if name.endswith(".de") or name.endswith(".de.gz"):
                    z.extract(name, out_dir)
                    de_file = os.path.join(out_dir, name)
                    print(f"  Extracted DE: {name}")
                elif name.endswith(".en") or name.endswith(".en.gz"):
                    z.extract(name, out_dir)
                    en_file = os.path.join(out_dir, name)
                    print(f"  Extracted EN: {name}")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return None, None

    return de_file, en_file


def count_lines(filepath):
    """Count lines in a text file."""
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Download OPUS corpora via API")
    parser.add_argument("--only", nargs="+", choices=list(CORPORA.keys()),
                        help="Download only these corpora")
    parser.add_argument("--list", action="store_true", help="List available corpora")
    parser.add_argument("--find-urls", action="store_true",
                        help="Just find URLs, don't download")
    args = parser.parse_args()

    if args.list:
        print("Available corpora:")
        for key, info in CORPORA.items():
            print(f"  {key:15s} -> {info['name']:15s} (domain: {info['domain']})")
        return

    to_download = args.only if args.only else list(CORPORA.keys())
    base_dir = r"C:\glosswerk\data\raw"

    print("=" * 60)
    print("GlossWerk - OPUS Corpus Downloader (API)")
    print("=" * 60)

    results = {}

    for i, key in enumerate(to_download, 1):
        info = CORPORA[key]
        print(f"\n[{i}/{len(to_download)}] --- {info['name']} ({info['domain']}) ---")

        url, version = find_download_url(info["name"])

        if not url:
            print(f"  SKIPPED: Could not find download URL")
            results[key] = {"status": "failed", "reason": "no URL"}
            continue

        if args.find_urls:
            results[key] = {"status": "url_found", "url": url, "version": version}
            continue

        corpus_dir = os.path.join(base_dir, key)
        os.makedirs(corpus_dir, exist_ok=True)

        de_file, en_file = download_and_extract(url, key, corpus_dir)

        if de_file and en_file:
            de_lines = count_lines(de_file)
            en_lines = count_lines(en_file)
            print(f"  DE lines: {de_lines:,}")
            print(f"  EN lines: {en_lines:,}")
            if de_lines == en_lines:
                print(f"  ALIGNED: {de_lines:,} sentence pairs")
            else:
                print(f"  WARNING: Line count mismatch!")
            results[key] = {"status": "ok", "pairs": min(de_lines, en_lines),
                           "de": de_file, "en": en_file}
        else:
            results[key] = {"status": "failed", "reason": "download/extract error"}

    print("\n" + "=" * 60)
    print("Summary:")
    for key, res in results.items():
        domain = CORPORA[key]["domain"]
        if res["status"] == "ok":
            print(f"  {key:15s} [{domain:10s}]: {res['pairs']:>10,} pairs")
        elif res["status"] == "url_found":
            print(f"  {key:15s} [{domain:10s}]: {res['url']}")
        else:
            print(f"  {key:15s} [{domain:10s}]: FAILED - {res.get('reason', 'unknown')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
