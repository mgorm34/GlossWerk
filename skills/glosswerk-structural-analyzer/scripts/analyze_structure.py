"""
GlossWerk Source-Side Structural Analyzer

Parses German patent sentences to compute structural risk features that
predict translation difficulty for DE→EN. Uses spaCy dependency parsing
to detect constructions that cause reordering problems in English:

1. Clause nesting depth — deeper = harder to restructure
2. Verb-final span — distance from subject to clause-final verb
3. Relative clause chains — stacked relative clauses
4. Participial constructions — extended participial attributes
5. Sentence length — baseline difficulty predictor
6. Pre-field length — tokens before the finite verb in main clause (Vorfeld)

Outputs a composite "reordering difficulty" score (0-1) per sentence.

Usage:
    python analyze_structure.py --input patent.docx --output analysis.json
    python analyze_structure.py --input patent.docx --output analysis.json --format tsv

Requires: spaCy + de_core_news_lg, python-docx
Install:  pip install spacy python-docx && python -m spacy download de_core_news_lg
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Text extraction & sentence splitting (shared logic with term scanner)
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
# Structural feature extraction (requires spaCy)
# ---------------------------------------------------------------------------

def load_spacy_model():
    """Load de_core_news_lg. Fail fast if unavailable."""
    try:
        import spacy
        nlp = spacy.load("de_core_news_lg")
        return nlp
    except (ImportError, OSError) as e:
        print(f"ERROR: spaCy with de_core_news_lg required for structural analysis.")
        print(f"Install: pip install spacy && python -m spacy download de_core_news_lg")
        print(f"Detail: {e}")
        sys.exit(1)


def get_clause_depth(token):
    """Compute nesting depth of a token by counting clause-marking ancestors."""
    depth = 0
    current = token
    while current.head != current:
        current = current.head
        # Count clause boundaries: subordinating conjunctions, relative pronouns
        if current.dep_ in ("rc", "cp", "sb") or current.pos_ == "SCONJ":
            depth += 1
    return depth


def compute_max_clause_depth(doc):
    """
    Maximum clause nesting depth in the sentence.
    Uses dependency labels to identify subordinate clause boundaries.

    German clause-marking deps in spaCy German model:
    - 'rc' = relative clause
    - 'cp' = complementizer phrase
    - 'mo' = modifier (often clausal)
    - 'oc' = clausal object
    - 'sb' = subject (can indicate embedded clause when verb is subordinate)
    """
    if len(doc) == 0:
        return 0

    # Strategy: count how many clause-introducing tokens are on the path
    # from any token to root. Clause introducers: SCONJ, relative pronouns,
    # and tokens whose dep signals a clause boundary.

    CLAUSE_DEPS = {"rc", "cp", "oc", "re", "cj"}
    CLAUSE_POS = {"SCONJ"}

    max_depth = 0
    for token in doc:
        depth = 0
        current = token
        visited = set()
        while current.head != current and current.i not in visited:
            visited.add(current.i)
            current = current.head
            if current.dep_ in CLAUSE_DEPS or current.pos_ in CLAUSE_POS:
                depth += 1
        if depth > max_depth:
            max_depth = depth

    return max_depth


def compute_verb_final_spans(doc):
    """
    Detect verb-final constructions and measure the span length.

    In German subordinate clauses, the finite verb goes to the end.
    Long spans between the subject/conjunction and the final verb are
    what cause the most reordering difficulty in EN translation.

    Returns: list of (start_idx, end_idx, span_length) for each verb-final span.
    """
    spans = []

    # Find subordinating conjunctions and relative pronouns that open clauses
    clause_openers = []
    for token in doc:
        if token.pos_ == "SCONJ":  # dass, wenn, weil, obwohl, etc.
            clause_openers.append(token)
        elif token.dep_ == "rc" and token.pos_ in ("PRON", "DET"):  # der, die, das as relative
            clause_openers.append(token)

    # For each clause opener, find the clause-final verb
    for opener in clause_openers:
        # Collect all tokens in this subordinate clause
        clause_tokens = _get_clause_tokens(doc, opener)
        if not clause_tokens:
            continue

        # Find the last verb in the clause
        verbs_in_clause = [t for t in clause_tokens if t.pos_ in ("VERB", "AUX")]
        if not verbs_in_clause:
            continue

        last_verb = max(verbs_in_clause, key=lambda t: t.i)
        span_length = last_verb.i - opener.i

        if span_length > 2:  # Only count non-trivial spans
            spans.append((opener.i, last_verb.i, span_length))

    return spans


def _get_clause_tokens(doc, opener):
    """
    Get all tokens belonging to the clause opened by this token.
    Walk right from the opener until we hit a token that's not a descendant
    or until we hit another clause boundary.
    """
    clause_tokens = [opener]

    # Collect descendants of the opener's head verb (the clause verb)
    # or tokens that are syntactically within this clause
    start_idx = opener.i

    # Simple heuristic: take tokens from opener to the next comma/period
    # or to the end of sentence, whichever comes first
    for token in doc[start_idx + 1:]:
        if token.text in (".", ";") or (token.text == "," and token.i > start_idx + 3):
            break
        clause_tokens.append(token)

    return clause_tokens


def count_relative_clauses(doc):
    """Count relative clause chains (stacked relative clauses)."""
    rc_count = 0
    for token in doc:
        # German relative pronouns: der/die/das/dem/den/dessen/deren/denen
        # when functioning as relative clause introducers
        if token.dep_ == "rc":
            rc_count += 1
        # Also catch "welch-" relatives
        elif token.text.lower().startswith("welch") and token.dep_ in ("rc", "sb", "oa"):
            rc_count += 1

    # Also count by pattern: comma + relative pronoun
    for i, token in enumerate(doc[1:], 1):
        if (doc[i-1].text == "," and
            token.text.lower() in ("der", "die", "das", "dem", "den",
                                    "dessen", "deren", "denen") and
            token.pos_ in ("PRON", "DET") and
            token.dep_ != "det"):
            rc_count = max(rc_count, 1)  # At least 1 if pattern matches

    return rc_count


def detect_participial_constructions(doc):
    """
    Detect extended participial attributes (erweiterte Partizipialattribute).

    These are a major source of DE→EN translation difficulty because they
    must be unpacked into relative clauses in English.

    Pattern: determiner ... participle ... noun
    Example: "die in dem Herzen angeordnete Klappe"
             → "the valve arranged in the heart"

    The participle (angeordnete) comes BEFORE the noun it modifies,
    with potentially many tokens between the determiner and the participle.
    """
    participials = []

    for token in doc:
        # Look for participles (past participles used as adjectives)
        # German past participles: ge-...-t, ge-...-en, or irregular
        if token.pos_ == "ADJ" and token.tag_ and "Part" in token.tag_:
            # This is a participial adjective
            # Find the noun it modifies
            head = token.head
            if head.pos_ in ("NOUN", "PROPN"):
                # Find the determiner for this noun
                det = None
                for child in head.children:
                    if child.dep_ == "det" or child.dep_ == "nk" and child.pos_ == "DET":
                        if child.i < token.i:
                            det = child
                            break

                if det:
                    span_length = token.i - det.i
                    if span_length > 2:  # Non-trivial participial construction
                        participials.append({
                            "det_idx": det.i,
                            "part_idx": token.i,
                            "noun_idx": head.i,
                            "span_length": span_length,
                            "text": doc[det.i:head.i + 1].text
                        })

        # Also catch present participles used attributively
        # Pattern: adjective ending in -end/-ende/-enden (present participle)
        if (token.pos_ == "ADJ" and
            any(token.text.lower().endswith(s) for s in ("end", "ende", "enden", "ender", "endes", "endem"))):
            head = token.head
            if head.pos_ in ("NOUN", "PROPN"):
                det = None
                for child in head.children:
                    if child.dep_ in ("det", "nk") and child.pos_ == "DET":
                        if child.i < token.i:
                            det = child
                            break
                if det:
                    span_length = token.i - det.i
                    if span_length > 2:
                        participials.append({
                            "det_idx": det.i,
                            "part_idx": token.i,
                            "noun_idx": head.i,
                            "span_length": span_length,
                            "text": doc[det.i:head.i + 1].text
                        })

    return participials


def compute_vorfeld_length(doc):
    """
    Compute pre-field (Vorfeld) length: tokens before the finite verb in main clause.

    German V2 rule: the finite verb is in second position in main clauses.
    A long Vorfeld (many tokens before the verb) indicates a topicalized
    construction that may need restructuring in English.

    Normal: "Der Stent weist einen Bügel auf." (Vorfeld = 2 tokens)
    Long:   "Der in dem Herzen des Patienten angeordnete selbstexpandierende
             Mitralklappenstent weist..." (Vorfeld = 8+ tokens)
    """
    # Find the main (root) verb
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None or root.pos_ not in ("VERB", "AUX"):
        return 0

    # Count tokens before the root verb
    vorfeld_length = root.i  # Tokens before the root

    return vorfeld_length


def compute_sentence_features(doc):
    """
    Compute all structural features for a single parsed sentence.
    Returns a dict of features.
    """
    n_tokens = len(doc)

    # Skip very short sentences (headers, labels)
    if n_tokens < 4:
        return {
            "n_tokens": n_tokens,
            "clause_depth": 0,
            "max_verb_final_span": 0,
            "n_verb_final_clauses": 0,
            "n_relative_clauses": 0,
            "n_participial_constructions": 0,
            "max_participial_span": 0,
            "vorfeld_length": 0,
            "risk_score": 0.0,
            "risk_level": "low",
            "risk_factors": [],
        }

    # Feature 1: Clause nesting depth
    clause_depth = compute_max_clause_depth(doc)

    # Feature 2: Verb-final spans
    vf_spans = compute_verb_final_spans(doc)
    max_vf_span = max((s[2] for s in vf_spans), default=0)
    n_vf_clauses = len(vf_spans)

    # Feature 3: Relative clauses
    n_rel_clauses = count_relative_clauses(doc)

    # Feature 4: Participial constructions
    participials = detect_participial_constructions(doc)
    n_participials = len(participials)
    max_part_span = max((p["span_length"] for p in participials), default=0)

    # Feature 5: Sentence length (token count)
    # Already have n_tokens

    # Feature 6: Vorfeld length
    vorfeld = compute_vorfeld_length(doc)

    # --- Composite risk score ---
    # Each feature contributes to the 0-1 score.
    # Weights reflect how much each feature predicts reordering difficulty
    # based on DE→EN translation error patterns.

    risk_factors = []
    score_components = []

    # Length component (0-0.25): longer sentences are harder
    # 10 tokens = 0, 30 tokens = 0.125, 50+ tokens = 0.25
    length_score = min(max(n_tokens - 10, 0) / 160, 0.25)
    score_components.append(length_score)
    if n_tokens > 35:
        risk_factors.append(f"long sentence ({n_tokens} tokens)")

    # Clause depth component (0-0.25): deeper nesting = harder
    # depth 0 = 0, depth 1 = 0.08, depth 2 = 0.17, depth 3+ = 0.25
    depth_score = min(clause_depth / 3, 1.0) * 0.25
    score_components.append(depth_score)
    if clause_depth >= 2:
        risk_factors.append(f"clause depth {clause_depth}")

    # Verb-final span component (0-0.2): long V-final spans = hard reordering
    # span 0 = 0, span 5 = 0.1, span 10+ = 0.2
    vf_score = min(max_vf_span / 10, 1.0) * 0.2
    score_components.append(vf_score)
    if max_vf_span > 5:
        risk_factors.append(f"verb-final span of {max_vf_span} tokens")

    # Relative clause component (0-0.1): stacked RCs = harder
    rc_score = min(n_rel_clauses / 2, 1.0) * 0.1
    score_components.append(rc_score)
    if n_rel_clauses >= 2:
        risk_factors.append(f"{n_rel_clauses} relative clauses")

    # Participial construction component (0-0.1): extended participials = restructuring needed
    part_score = min(n_participials / 2, 1.0) * 0.1
    if max_part_span > 4:
        part_score = min(part_score + 0.05, 0.1)
    score_components.append(part_score)
    if n_participials > 0:
        risk_factors.append(f"{n_participials} participial construction(s)")

    # Vorfeld component (0-0.1): long pre-field = topicalization issues
    vf_field_score = min(max(vorfeld - 3, 0) / 7, 1.0) * 0.1
    score_components.append(vf_field_score)
    if vorfeld > 5:
        risk_factors.append(f"long Vorfeld ({vorfeld} tokens)")

    # Composite score
    risk_score = round(sum(score_components), 3)

    # Risk level thresholds
    if risk_score >= 0.5:
        risk_level = "high"
    elif risk_score >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "n_tokens": n_tokens,
        "clause_depth": clause_depth,
        "max_verb_final_span": max_vf_span,
        "n_verb_final_clauses": n_vf_clauses,
        "n_relative_clauses": n_rel_clauses,
        "n_participial_constructions": n_participials,
        "max_participial_span": max_part_span,
        "vorfeld_length": vorfeld,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
    }


# ---------------------------------------------------------------------------
# Translation instruction generation
# ---------------------------------------------------------------------------

def generate_translation_hint(features):
    """
    Generate a per-sentence translation instruction based on structural features.
    This gets injected into the LLM translation prompt for high-risk segments.
    """
    if features["risk_level"] == "low":
        return None

    hints = []

    if features["clause_depth"] >= 2:
        hints.append(
            f"This sentence has {features['clause_depth']}-level clause nesting. "
            "Restructure for English by unpacking subordinate clauses — "
            "place the most important information earlier in the sentence."
        )

    if features["max_verb_final_span"] > 5:
        hints.append(
            f"Contains a verb-final clause spanning {features['max_verb_final_span']} tokens. "
            "The key verb/predicate arrives late in German — "
            "front-load it in the English translation."
        )

    if features["n_relative_clauses"] >= 2:
        hints.append(
            f"Contains {features['n_relative_clauses']} stacked relative clauses. "
            "Consider breaking into separate sentences or restructuring "
            "to avoid nested 'which...that...' chains in English."
        )

    if features["n_participial_constructions"] > 0:
        hints.append(
            f"Contains {features['n_participial_constructions']} extended participial "
            f"attribute(s) (span up to {features['max_participial_span']} tokens). "
            "Convert to relative clause(s) in English: "
            "'the X arranged in Y' → 'the X that is arranged in Y'."
        )

    if features["vorfeld_length"] > 5:
        hints.append(
            f"Long pre-field ({features['vorfeld_length']} tokens before the main verb). "
            "The topicalized element may need to be repositioned in English "
            "for natural information flow."
        )

    if not hints and features["n_tokens"] > 35:
        hints.append(
            f"Long sentence ({features['n_tokens']} tokens). "
            "Pay attention to constituent order — "
            "English should front-load key information."
        )

    return " ".join(hints) if hints else None


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_document(docx_path, nlp=None):
    """
    Full structural analysis of a German patent document.

    Returns:
        dict with:
            - sentences: list of {text, index, features, hint}
            - summary: aggregate statistics
    """
    if nlp is None:
        nlp = load_spacy_model()

    # Extract and split
    raw_text = extract_text_from_docx(docx_path)
    sentences = split_sentences(raw_text)

    results = []
    risk_counts = Counter()
    total_score = 0.0

    for i, sent_text in enumerate(sentences):
        doc = nlp(sent_text)
        features = compute_sentence_features(doc)
        hint = generate_translation_hint(features)

        results.append({
            "index": i,
            "text": sent_text,
            "features": features,
            "translation_hint": hint,
        })

        risk_counts[features["risk_level"]] += 1
        total_score += features["risk_score"]

    n_sentences = len(sentences)
    avg_score = round(total_score / n_sentences, 3) if n_sentences > 0 else 0

    summary = {
        "total_sentences": n_sentences,
        "avg_risk_score": avg_score,
        "risk_distribution": {
            "high": risk_counts.get("high", 0),
            "medium": risk_counts.get("medium", 0),
            "low": risk_counts.get("low", 0),
        },
        "high_risk_pct": round(risk_counts.get("high", 0) / n_sentences * 100, 1) if n_sentences else 0,
        "avg_tokens": round(sum(r["features"]["n_tokens"] for r in results) / n_sentences, 1) if n_sentences else 0,
        "max_clause_depth": max((r["features"]["clause_depth"] for r in results), default=0),
        "sentences_with_participials": sum(1 for r in results if r["features"]["n_participial_constructions"] > 0),
        "sentences_with_relative_clauses": sum(1 for r in results if r["features"]["n_relative_clauses"] > 0),
    }

    return {
        "sentences": results,
        "summary": summary,
    }


def format_as_tsv(analysis):
    """Format analysis as TSV for quick review."""
    lines = [
        "idx\trisk_score\trisk_level\ttokens\tclause_depth\tvf_span\trel_clauses\tparticipials\tvorfeld\trisk_factors\ttext_preview"
    ]
    for sent in analysis["sentences"]:
        f = sent["features"]
        preview = sent["text"][:80].replace("\t", " ")
        factors = "; ".join(f["risk_factors"]) if f["risk_factors"] else "-"
        lines.append(
            f"{sent['index']}\t{f['risk_score']}\t{f['risk_level']}\t{f['n_tokens']}\t"
            f"{f['clause_depth']}\t{f['max_verb_final_span']}\t{f['n_relative_clauses']}\t"
            f"{f['n_participial_constructions']}\t{f['vorfeld_length']}\t{factors}\t{preview}"
        )
    return "\n".join(lines)


def print_summary(analysis, filename=""):
    """Print a human-readable summary to stdout."""
    s = analysis["summary"]
    print(f"\n{'='*60}")
    print(f"STRUCTURAL ANALYSIS — {filename}")
    print(f"{'='*60}")
    print(f"Total sentences: {s['total_sentences']}")
    print(f"Average risk score: {s['avg_risk_score']:.3f}")
    print(f"Average tokens/sentence: {s['avg_tokens']}")
    print(f"Max clause depth found: {s['max_clause_depth']}")
    print()
    print(f"Risk distribution:")
    print(f"  HIGH   (≥0.5):  {s['risk_distribution']['high']:3d}  ({s['high_risk_pct']:.1f}%)")
    print(f"  MEDIUM (0.25-0.5): {s['risk_distribution']['medium']:3d}")
    print(f"  LOW    (<0.25): {s['risk_distribution']['low']:3d}")
    print()
    print(f"Sentences with participial constructions: {s['sentences_with_participials']}")
    print(f"Sentences with relative clauses: {s['sentences_with_relative_clauses']}")
    print()

    # Show top 5 highest-risk sentences
    high_risk = sorted(analysis["sentences"], key=lambda x: x["features"]["risk_score"], reverse=True)[:5]
    if high_risk and high_risk[0]["features"]["risk_score"] > 0:
        print(f"TOP 5 HIGHEST-RISK SENTENCES:")
        print(f"{'-'*60}")
        for sent in high_risk:
            f = sent["features"]
            preview = sent["text"][:100]
            if len(sent["text"]) > 100:
                preview += "..."
            print(f"  [{sent['index']}] score={f['risk_score']:.3f} ({f['risk_level']})")
            print(f"       {preview}")
            if f["risk_factors"]:
                print(f"       Factors: {', '.join(f['risk_factors'])}")
            if sent["translation_hint"]:
                print(f"       Hint: {sent['translation_hint'][:120]}")
            print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Structural analysis of German patent documents")
    parser.add_argument("--input", required=True, help="Path to German .docx file")
    parser.add_argument("--output", required=True, help="Output file path (.json or .tsv)")
    parser.add_argument("--format", choices=["json", "tsv"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    nlp = load_spacy_model()
    analysis = analyze_document(str(input_path), nlp)

    # Save output
    output_path = Path(args.output)
    if args.format == "tsv":
        output_path.write_text(format_as_tsv(analysis), encoding="utf-8")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print_summary(analysis, input_path.name)
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
