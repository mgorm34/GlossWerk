"""
GlossWerk LLM Translator Module

Translates German patent documents to English using Claude, with:
1. Document-level context for terminology consistency
2. Per-sentence structural hints from the structural analyzer
3. Glossary enforcement (user-provided or from term scanner)
4. Batched API calls for long documents

This module is imported by the integrated Streamlit app.
It can also be run standalone for testing:
    python translate.py --input patent.docx --output translations.json \
        --api-key YOUR_KEY --structural-analysis analysis.json --glossary glossary.tsv

Requires: anthropic, python-docx
Install:  pip install anthropic python-docx
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from prompt_layers import build_translation_prompt


# ---------------------------------------------------------------------------
# System prompt (assembled from prompt_layers module)
# ---------------------------------------------------------------------------

def build_system_prompt(glossary=None, custom_instructions=None, domain="patent"):
    """
    Build the full system prompt with optional glossary and custom instructions.

    Delegates to prompt_layers.build_translation_prompt() for layered assembly:
      Tier 1 — Core DE→EN linguistics (domain-agnostic)
      Tier 2 — Domain overlay (patent, general, etc.)
      Tier 3 — Glossary + custom instructions

    Args:
        glossary: dict of {german_term: english_translation} or None
        custom_instructions: str of additional instructions or None
        domain: str — "patent", "general", etc.
    Returns:
        Complete system prompt string
    """
    return build_translation_prompt(
        domain=domain,
        glossary=glossary,
        custom_instructions=custom_instructions,
    )


# ---------------------------------------------------------------------------
# Structural hint injection
# ---------------------------------------------------------------------------

def build_annotated_input(sentences, structural_analysis=None):
    """
    Build the numbered input for translation, injecting structural hints
    for medium/high-risk sentences.

    Args:
        sentences: list of German sentence strings
        structural_analysis: dict from analyze_structure.py output, or None

    Returns:
        str — numbered input with inline structural hints
    """
    # Index structural data by sentence index
    hints_by_idx = {}
    if structural_analysis and "sentences" in structural_analysis:
        for entry in structural_analysis["sentences"]:
            idx = entry.get("index")
            hint = entry.get("translation_hint")
            risk = entry.get("features", {}).get("risk_level", "low")
            if hint and risk in ("medium", "high"):
                hints_by_idx[idx] = {
                    "hint": hint,
                    "risk": risk,
                    "score": entry.get("features", {}).get("risk_score", 0),
                }

    lines = []
    for i, sent in enumerate(sentences):
        if i in hints_by_idx:
            h = hints_by_idx[i]
            # Inject hint as a bracketed instruction before the sentence
            lines.append(
                f"[{i+1}] {{STRUCTURAL NOTE (risk={h['risk']}, score={h['score']:.2f}): "
                f"{h['hint']}}} {sent}"
            )
        else:
            lines.append(f"[{i+1}] {sent}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Translation API calls
# ---------------------------------------------------------------------------

def get_client(api_key):
    """Create Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def translate_document(sentences, api_key, model="claude-sonnet-4-6",
                       glossary=None, structural_analysis=None,
                       custom_instructions=None,
                       batch_size=50, progress_callback=None,
                       domain="patent"):
    """
    Translate a full German patent document.

    For documents up to batch_size sentences, sends everything in one call
    for maximum terminology consistency. For longer documents, batches with
    overlap to maintain context.

    Args:
        sentences: list of German sentences
        api_key: Anthropic API key
        model: Claude model name
        glossary: dict {DE: EN} or None
        structural_analysis: output from analyze_structure.py or None
        custom_instructions: additional prompt text or None
        batch_size: max sentences per API call
        progress_callback: fn(current, total) for progress updates

    Returns:
        list of dicts: [{
            "index": 0,
            "source": "German sentence",
            "translation": "English sentence",
            "risk_level": "low|medium|high",
            "risk_score": 0.0,
            "had_structural_hint": False
        }, ...]
    """
    system_prompt = build_system_prompt(glossary, custom_instructions, domain=domain)
    n = len(sentences)

    # Build structural lookup
    risk_lookup = {}
    if structural_analysis and "sentences" in structural_analysis:
        for entry in structural_analysis["sentences"]:
            idx = entry.get("index")
            risk_lookup[idx] = {
                "risk_level": entry.get("features", {}).get("risk_level", "low"),
                "risk_score": entry.get("features", {}).get("risk_score", 0.0),
                "had_hint": entry.get("translation_hint") is not None
                            and entry.get("features", {}).get("risk_level") in ("medium", "high"),
            }

    # Single batch if small enough
    if n <= batch_size:
        translations = _translate_batch(
            sentences, 0, api_key, model, system_prompt, structural_analysis
        )
        if progress_callback:
            progress_callback(n, n)
    else:
        # Multi-batch with 5-sentence overlap for context continuity
        translations = [""] * n
        overlap = 5
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            batch_sents = sentences[start:end]

            # Build structural analysis subset for this batch
            batch_analysis = _subset_analysis(structural_analysis, start, end)

            batch_translations = _translate_batch(
                batch_sents, start, api_key, model, system_prompt, batch_analysis
            )

            # For overlap region, keep the translation from the batch that
            # had more context (i.e., don't overwrite earlier batch's translations
            # in the overlap zone unless this is the first batch)
            write_start = 0 if start == 0 else overlap
            for j in range(write_start, len(batch_translations)):
                global_idx = start + j
                if global_idx < n:
                    translations[global_idx] = batch_translations[j]

            if progress_callback:
                progress_callback(min(end, n), n)

            # If we've reached the end, stop
            if end >= n:
                break

            # Next batch starts overlap tokens before the end of this one
            start = end - overlap

    # Assemble results
    results = []
    for i, sent in enumerate(sentences):
        rl = risk_lookup.get(i, {})
        results.append({
            "index": i,
            "source": sent,
            "translation": translations[i] if i < len(translations) else "[MISSING]",
            "risk_level": rl.get("risk_level", "unknown"),
            "risk_score": rl.get("risk_score", 0.0),
            "had_structural_hint": rl.get("had_hint", False),
        })

    return results


def _subset_analysis(structural_analysis, start, end):
    """Extract a subset of structural analysis for a batch range, re-indexing."""
    if not structural_analysis or "sentences" not in structural_analysis:
        return None

    subset = {"sentences": []}
    for entry in structural_analysis["sentences"]:
        idx = entry.get("index", -1)
        if start <= idx < end:
            # Re-index relative to batch start
            new_entry = dict(entry)
            new_entry["index"] = idx - start
            subset["sentences"].append(new_entry)

    return subset if subset["sentences"] else None


def _translate_batch(sentences, global_offset, api_key, model,
                     system_prompt, structural_analysis):
    """
    Translate a batch of sentences in one API call.

    Args:
        sentences: list of German sentences for this batch
        global_offset: the starting index in the full document (for structural lookup)
        api_key: Anthropic API key
        model: model name
        system_prompt: full system prompt
        structural_analysis: structural analysis dict (already subset/re-indexed for batch)

    Returns:
        list of English translations
    """
    client = get_client(api_key)

    # Build input with structural annotations
    annotated_input = build_annotated_input(sentences, structural_analysis)

    user_message = (
        "Translate each numbered German patent sentence below into English. "
        "Return ONLY the translations as a numbered list in the same format: "
        "[1] translation, [2] translation, etc. "
        "One English sentence per German sentence. "
        "Preserve all paragraph reference numbers like [0012] within the text. "
        "Do not add commentary, notes, or explanations.\n\n"
        "Some sentences have a {STRUCTURAL NOTE} before them — these describe "
        "specific structural difficulties in the German source. Use these notes "
        "to guide your restructuring of those sentences for natural English. "
        "Do NOT include the structural notes in your translations.\n\n"
        f"{annotated_input}"
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        if not message.content or len(message.content) == 0:
            return ["[EMPTY RESPONSE]"] * len(sentences)

        raw = message.content[0].text.strip()
        return parse_numbered_output(raw, len(sentences))

    except Exception as e:
        print(f"ERROR in translation batch: {e}", file=sys.stderr)
        # Fallback: try sentence-by-sentence
        return _translate_fallback(sentences, client, model, system_prompt)


def _translate_fallback(sentences, client, model, system_prompt):
    """Sentence-by-sentence fallback if batch call fails."""
    translations = []
    for sent in sentences:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Translate this German patent text to English:\n\n{sent}"
                }],
            )
            if message.content:
                translations.append(message.content[0].text.strip())
            else:
                translations.append("[EMPTY RESPONSE]")
        except Exception as e:
            translations.append(f"[ERROR: {e}]")
        time.sleep(0.5)  # Rate limiting
    return translations


# ---------------------------------------------------------------------------
# Output parsing (shared with hter_training_builder)
# ---------------------------------------------------------------------------

def parse_numbered_output(raw_text, expected_count):
    """Parse Claude's numbered translation output into a list."""
    # Try [N] format first
    pattern_bracket = re.findall(
        r'\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)', raw_text, re.DOTALL
    )
    if len(pattern_bracket) >= expected_count * 0.8:
        result = [""] * expected_count
        for num_str, text in pattern_bracket:
            idx = int(num_str) - 1
            if 0 <= idx < expected_count:
                result[idx] = text.strip()
        for i in range(expected_count):
            if not result[i]:
                result[i] = "[PARSE ERROR — review manually]"
        return result

    # Try N. format
    pattern_dot = re.findall(
        r'(\d+)\.\s+(.+?)(?=\n\d+\.|\Z)', raw_text, re.DOTALL
    )
    if len(pattern_dot) >= expected_count * 0.8:
        result = [""] * expected_count
        for num_str, text in pattern_dot:
            idx = int(num_str) - 1
            if 0 <= idx < expected_count:
                result[idx] = text.strip()
        for i in range(expected_count):
            if not result[i]:
                result[i] = "[PARSE ERROR — review manually]"
        return result

    # Fallback: split by lines
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r'^\[\d+\]\s*', '', line)
        line = re.sub(r'^\d+\.\s+', '', line)
        if line:
            cleaned.append(line)

    while len(cleaned) < expected_count:
        cleaned.append("[PARSE ERROR — review manually]")

    return cleaned[:expected_count]


# ---------------------------------------------------------------------------
# Text extraction & sentence splitting (shared)
# ---------------------------------------------------------------------------

def extract_text_from_docx(filepath):
    """Extract text from a .docx file."""
    from docx import Document as DocxDocument
    doc = DocxDocument(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text and len(text) > 10:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def split_sentences(text):
    """Split German patent text into sentences, protecting abbreviations."""
    if not text or not text.strip():
        return []
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    protected = {
        'z.B.': 'Z__B__', 'z. B.': 'Z__B__',
        'd.h.': 'D__H__', 'd. h.': 'D__H__',
        'Fig.': 'FIG__', 'fig.': 'FIG__',
        'Nr.': 'NR__', 'Abs.': 'ABS__',
        'bzw.': 'BZW__', 'ca.': 'CA__',
        'etc.': 'ETC__', 'evtl.': 'EVTL__',
        'ggf.': 'GGF__', 'inkl.': 'INKL__',
        'max.': 'MAX__', 'min.': 'MIN__',
        'sog.': 'SOG__', 'u.a.': 'U__A__',
        'vgl.': 'VGL__', 'vol.': 'VOL__',
    }

    for orig, placeholder in protected.items():
        text = text.replace(orig, placeholder)

    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ\[(])', text)

    sentences = []
    for part in parts:
        for orig, placeholder in protected.items():
            part = part.replace(placeholder, orig)
        part = part.strip()
        if len(part) > 15:
            sentences.append(part)
    return sentences


# ---------------------------------------------------------------------------
# Glossary loading
# ---------------------------------------------------------------------------

def load_glossary_tsv(filepath):
    """Load glossary from TSV file (DE<tab>EN per line)."""
    glossary = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                glossary[parts[0].strip()] = parts[1].strip()
    return glossary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Translate German patent with structural hints")
    parser.add_argument("--input", required=True, help="German .docx file")
    parser.add_argument("--output", required=True, help="Output .json file")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model")
    parser.add_argument("--structural-analysis", default=None,
                        help="JSON from analyze_structure.py")
    parser.add_argument("--glossary", default=None, help="Glossary TSV file")
    parser.add_argument("--batch-size", type=int, default=50, help="Sentences per API call")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set ANTHROPIC_API_KEY")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)

    # Load structural analysis if provided
    structural = None
    if args.structural_analysis:
        sa_path = Path(args.structural_analysis)
        if sa_path.exists():
            with open(sa_path, "r", encoding="utf-8") as f:
                structural = json.load(f)
            print(f"Loaded structural analysis: {len(structural.get('sentences', []))} sentences")
        else:
            print(f"WARNING: Structural analysis not found: {sa_path}")

    # Load glossary if provided
    glossary = None
    if args.glossary:
        g_path = Path(args.glossary)
        if g_path.exists():
            glossary = load_glossary_tsv(str(g_path))
            print(f"Loaded glossary: {len(glossary)} terms")
        else:
            print(f"WARNING: Glossary not found: {g_path}")

    # Extract and split
    raw_text = extract_text_from_docx(str(input_path))
    sentences = split_sentences(raw_text)
    print(f"Document: {len(sentences)} sentences")

    # Translate
    def progress(current, total):
        print(f"  Translated {current}/{total} sentences...")

    results = translate_document(
        sentences=sentences,
        api_key=api_key,
        model=args.model,
        glossary=glossary,
        structural_analysis=structural,
        batch_size=args.batch_size,
        progress_callback=progress,
    )

    # Count hints used
    hinted = sum(1 for r in results if r["had_structural_hint"])
    print(f"\nDone. {hinted}/{len(results)} sentences had structural hints.")

    # Risk breakdown
    from collections import Counter
    risk_counts = Counter(r["risk_level"] for r in results)
    print(f"Risk distribution: {dict(risk_counts)}")

    # Save
    output = {
        "metadata": {
            "source_file": str(input_path),
            "model": args.model,
            "n_sentences": len(sentences),
            "n_structural_hints": hinted,
            "glossary_terms": len(glossary) if glossary else 0,
        },
        "translations": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
