"""
GlossWerk Prompt Evolver — Self-Improving Prompt Pipeline

Analyzes translator feedback to detect recurring edit patterns,
proposes new prompt rules, and stages them for human approval + A/B eval.

Architecture:
  Stage 1 — DETECT:  Classify each edit by type (terminology, reordering, etc.)
                      using lightweight string diff analysis.
  Stage 2 — CLUSTER: Aggregate edit types across all feedback. Flag patterns
                      that appear 5+ times as "recurring misses."
  Stage 3 — PROPOSE: Feed recurring-miss clusters to Claude API and ask it to
                      draft a candidate prompt rule with examples.
  Stage 4 — REVIEW:  Human reviews proposal, runs A/B eval, approves or rejects.

The proposals are stored in data/prompt_proposals/ as JSON files.
Nothing is injected into prompt_layers.py automatically.

Usage:
    # Analyze feedback and generate proposals
    python prompt_evolver.py --analyze --api-key KEY

    # List pending proposals
    python prompt_evolver.py --list-proposals

    # Apply an approved proposal to prompt_layers.py
    python prompt_evolver.py --apply proposal_20260327_001.json
"""

import json
import os
import re
import sys
import difflib
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_data_root = os.environ.get("DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
FEEDBACK_FILE = os.path.join(_data_root, "feedback_pairs.jsonl")
TRAINING_FILE = os.path.join(_data_root, "hter_training", "training_pairs.jsonl")
PROPOSALS_DIR = os.path.join(_data_root, "prompt_proposals")


# ---------------------------------------------------------------------------
# Stage 1: Edit-type classification
# ---------------------------------------------------------------------------

# Patterns that detect specific edit types from the diff between original and edited
EDIT_PATTERNS = {
    "terminology_swap": {
        "description": "Single term replaced with different term, rest unchanged",
        "detect": "_detect_terminology_swap",
    },
    "nominalization_to_verb": {
        "description": "Nominalization (the X-ation of Y) converted to verbal construction",
        "detect": "_detect_nominalization_fix",
    },
    "word_order_restructure": {
        "description": "Significant word reordering (information structure fix)",
        "detect": "_detect_word_order_change",
    },
    "genitive_chain_fix": {
        "description": "'X of the Y' converted to possessive or compound",
        "detect": "_detect_genitive_fix",
    },
    "passive_to_active": {
        "description": "Passive construction converted to active voice",
        "detect": "_detect_passive_to_active",
    },
    "calque_fix": {
        "description": "German syntactic calque rewritten in natural English",
        "detect": "_detect_calque_fix",
    },
    "addition": {
        "description": "Words added that weren't in original translation",
        "detect": "_detect_addition",
    },
    "deletion": {
        "description": "Words removed from original translation",
        "detect": "_detect_deletion",
    },
}


def classify_edit(original: str, edited: str, source_de: str = "") -> dict:
    """
    Classify what type of edit a translator made.

    Returns:
        {
            "edit_types": ["terminology_swap", ...],
            "primary_type": "terminology_swap",
            "diff_ratio": 0.85,  # similarity ratio
            "changed_words": ["Befestigung", "fastening"],
            "details": "Replaced 'the fastening of' with 'fastening'"
        }
    """
    if original.strip() == edited.strip():
        return {"edit_types": [], "primary_type": "none", "diff_ratio": 1.0,
                "changed_words": [], "details": "No changes"}

    # Compute basic diff metrics
    ratio = difflib.SequenceMatcher(None, original.lower(), edited.lower()).ratio()
    orig_words = original.split()
    edit_words = edited.split()
    orig_set = set(w.lower().strip(".,;:()") for w in orig_words)
    edit_set = set(w.lower().strip(".,;:()") for w in edit_words)

    added_words = edit_set - orig_set
    removed_words = orig_set - edit_set

    detected_types = []
    details_parts = []

    # Run each detector
    if _detect_terminology_swap(original, edited, orig_words, edit_words, added_words, removed_words):
        detected_types.append("terminology_swap")
        details_parts.append(f"Term swap: removed {removed_words & (orig_set - edit_set)}, added {added_words & (edit_set - orig_set)}")

    if _detect_nominalization_fix(original, edited, orig_words, edit_words):
        detected_types.append("nominalization_to_verb")
        details_parts.append("Nominalization converted to verbal construction")

    if _detect_word_order_change(original, edited, orig_words, edit_words):
        detected_types.append("word_order_restructure")
        details_parts.append("Significant word reordering")

    if _detect_genitive_fix(original, edited):
        detected_types.append("genitive_chain_fix")
        details_parts.append("Genitive 'of' chain simplified")

    if _detect_passive_to_active(original, edited):
        detected_types.append("passive_to_active")
        details_parts.append("Passive → active voice conversion")

    if _detect_calque_fix(original, edited, source_de):
        detected_types.append("calque_fix")
        details_parts.append("German syntactic calque rewritten")

    if not detected_types:
        # Fallback classification based on diff size
        if len(added_words) > 2 and len(removed_words) <= 1:
            detected_types.append("addition")
            details_parts.append(f"Added: {added_words}")
        elif len(removed_words) > 2 and len(added_words) <= 1:
            detected_types.append("deletion")
            details_parts.append(f"Removed: {removed_words}")
        else:
            detected_types.append("general_rephrase")
            details_parts.append("General rephrasing")

    return {
        "edit_types": detected_types,
        "primary_type": detected_types[0] if detected_types else "unknown",
        "diff_ratio": round(ratio, 3),
        "changed_words": list(added_words | removed_words)[:20],
        "details": "; ".join(details_parts),
    }


def _detect_terminology_swap(orig, edited, orig_words, edit_words, added, removed):
    """Detect single-term replacement: few words changed, rest is identical."""
    if len(added) <= 3 and len(removed) <= 3 and len(added) >= 1 and len(removed) >= 1:
        # Check that most of the sentence is unchanged
        ratio = difflib.SequenceMatcher(None, orig.lower(), edited.lower()).ratio()
        if ratio > 0.75:
            return True
    return False


def _detect_nominalization_fix(orig, edited, orig_words, edit_words):
    """Detect conversion of 'the X-ation of Y' to a verbal construction."""
    # Common nominalization patterns in original
    nom_patterns = [
        r'the\s+\w+(?:tion|ment|ance|ence|ing)\s+of\b',
        r'(?:facilitate|enable|effect|perform|carry out)\s+the\s+\w+(?:tion|ment|ance|ence)',
    ]
    orig_has_nom = any(re.search(p, orig, re.IGNORECASE) for p in nom_patterns)

    # Check if edited version removed the nominalization
    edit_has_nom = any(re.search(p, edited, re.IGNORECASE) for p in nom_patterns)

    return orig_has_nom and not edit_has_nom


def _detect_word_order_change(orig, edited, orig_words, edit_words):
    """Detect significant reordering (not just term swap)."""
    if len(orig_words) < 5 or len(edit_words) < 5:
        return False

    # Calculate word overlap
    common = set(w.lower() for w in orig_words) & set(w.lower() for w in edit_words)
    if len(common) < 4:
        return False

    # Check if common words appear in different order
    orig_positions = {w.lower(): i for i, w in enumerate(orig_words)}
    edit_positions = {w.lower(): i for i, w in enumerate(edit_words)}

    inversions = 0
    common_list = sorted(common)
    for i, w1 in enumerate(common_list):
        for w2 in common_list[i+1:]:
            if w1 in orig_positions and w2 in orig_positions and \
               w1 in edit_positions and w2 in edit_positions:
                orig_order = orig_positions[w1] < orig_positions[w2]
                edit_order = edit_positions[w1] < edit_positions[w2]
                if orig_order != edit_order:
                    inversions += 1

    # Significant reordering = many inversions relative to sentence length
    return inversions >= 3


def _detect_genitive_fix(orig, edited):
    """Detect 'X of the Y' → possessive or compound conversion."""
    of_pattern = r'\b(\w+)\s+of\s+the\s+(\w+)\b'
    orig_matches = len(re.findall(of_pattern, orig, re.IGNORECASE))
    edit_matches = len(re.findall(of_pattern, edited, re.IGNORECASE))
    return orig_matches > edit_matches and orig_matches >= 1


def _detect_passive_to_active(orig, edited):
    """Detect passive → active voice conversion."""
    passive_pattern = r'\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en|t)\b'
    orig_passive = len(re.findall(passive_pattern, orig, re.IGNORECASE))
    edit_passive = len(re.findall(passive_pattern, edited, re.IGNORECASE))
    return orig_passive > edit_passive and orig_passive >= 2


def _detect_calque_fix(orig, edited, source_de=""):
    """Detect German syntactic calque being rewritten to natural English."""
    calque_indicators = [
        # Funktionsverbgefüge calques
        r'comes?\s+to\s+(?:application|use)',
        r'stands?\s+in\s+(?:engagement|connection)',
        r'finds?\s+(?:use|application)',
        r'(?:is|are)\s+(?:effected|carried out)\s+by\s+(?:the\s+fact\s+that|means\s+of)',
        # Prenominal participial calques
        r'\b\w+\s+(?:\w+able|\w+ible)\s+by\s+(?:the|a)\b',
        # "in that" for "indem"
        r'\bin\s+that\b',
    ]
    orig_has_calque = any(re.search(p, orig, re.IGNORECASE) for p in calque_indicators)
    edit_has_calque = any(re.search(p, edited, re.IGNORECASE) for p in calque_indicators)

    return orig_has_calque and not edit_has_calque


# ---------------------------------------------------------------------------
# Stage 2: Pattern aggregation and disagreement detection
# ---------------------------------------------------------------------------

def load_feedback(path=None):
    """Load all feedback entries."""
    path = path or FEEDBACK_FILE
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_all_feedback(feedback_path=None, training_path=None):
    """Load feedback from both the pending file and the merged training file."""
    entries = load_feedback(feedback_path)

    # Also load user_feedback entries from training file
    tr_path = training_path or TRAINING_FILE
    if os.path.exists(tr_path):
        with open(tr_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("source") == "user_feedback":
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
    return entries


def analyze_feedback(entries=None):
    """
    Analyze all feedback to find recurring edit patterns and QE disagreements.

    Returns:
        {
            "total_entries": 150,
            "changed_entries": 45,
            "unchanged_entries": 105,
            "edit_type_counts": {"terminology_swap": 12, "nominalization_to_verb": 8, ...},
            "qe_disagreements": {
                "false_positives": [...],  # QE flagged but translator confirmed without edit
                "false_negatives": [...],  # QE said good but translator edited
            },
            "recurring_patterns": [
                {
                    "type": "nominalization_to_verb",
                    "count": 8,
                    "examples": [...],  # up to 10 representative examples
                    "qe_caught_count": 2,  # how many times QE flagged this correctly
                    "qe_missed_count": 6,  # how many times QE missed it
                }
            ],
        }
    """
    if entries is None:
        entries = load_all_feedback()

    changed = [e for e in entries if e.get("changed")]
    unchanged = [e for e in entries if not e.get("changed")]

    # Classify each edit
    classified = []
    for entry in changed:
        orig = entry.get("deepl", "")
        edited = entry.get("corrected", "")
        source = entry.get("de", "")
        classification = classify_edit(orig, edited, source)
        classified.append({**entry, "classification": classification})

    # Count edit types
    type_counts = Counter()
    type_examples = defaultdict(list)
    for c in classified:
        for et in c["classification"]["edit_types"]:
            type_counts[et] += 1
            if len(type_examples[et]) < 15:
                type_examples[et].append({
                    "de": c.get("de", "")[:300],
                    "original": c.get("deepl", "")[:300],
                    "edited": c.get("corrected", "")[:300],
                    "qe_rating": c.get("qe_rating_original", "unknown"),
                    "qe_category": c.get("qe_category_original", ""),
                    "details": c["classification"]["details"],
                })

    # QE disagreements
    false_positives = []  # QE flagged (minor/major/critical) but translator confirmed without edit
    false_negatives = []  # QE said "good" but translator edited

    for entry in unchanged:
        qe_rating = entry.get("qe_rating_original", "unknown")
        if qe_rating in ("major", "critical"):
            false_positives.append({
                "de": entry.get("de", "")[:300],
                "translation": entry.get("deepl", "")[:300],
                "qe_rating": qe_rating,
                "qe_category": entry.get("qe_category_original", ""),
            })

    for c in classified:
        if c.get("qe_rating_original") == "good":
            false_negatives.append({
                "de": c.get("de", "")[:300],
                "original": c.get("deepl", "")[:300],
                "edited": c.get("corrected", "")[:300],
                "edit_type": c["classification"]["primary_type"],
                "details": c["classification"]["details"],
            })

    # Build recurring patterns (5+ occurrences)
    recurring = []
    for edit_type, count in type_counts.most_common():
        if count < 3:  # lowered threshold for early use — raise to 5 once more data
            continue

        # How often did QE catch this vs miss it?
        examples = type_examples[edit_type]
        qe_caught = sum(1 for ex in examples if ex["qe_rating"] != "good")
        qe_missed = sum(1 for ex in examples if ex["qe_rating"] == "good")

        recurring.append({
            "type": edit_type,
            "count": count,
            "examples": examples[:10],
            "qe_caught_count": qe_caught,
            "qe_missed_count": qe_missed,
            "catch_rate": round(qe_caught / max(count, 1) * 100, 1),
        })

    return {
        "total_entries": len(entries),
        "changed_entries": len(changed),
        "unchanged_entries": len(unchanged),
        "edit_type_counts": dict(type_counts),
        "qe_disagreements": {
            "false_positives": false_positives[:20],
            "false_negatives": false_negatives[:20],
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
        },
        "recurring_patterns": recurring,
    }


# ---------------------------------------------------------------------------
# Stage 3: Rule proposal generation
# ---------------------------------------------------------------------------

RULE_PROPOSAL_SYSTEM = """\
You are an expert in DE→EN translation quality estimation prompt engineering.

You will be given a cluster of translator edits that represent a recurring pattern \
the current QE prompt is not catching. Your job is to draft a NEW prompt rule that \
would help the QE model detect this pattern.

CONSTRAINTS:
- The rule must be specific and actionable — not vague ("watch for errors")
- Include 2-3 concrete before/after examples from the provided data
- Specify which existing section of the QE prompt it belongs in (or if it's new)
- Specify the recommended severity rating for this pattern
- Keep the rule under 200 words
- Use the same format as existing rules in the prompt (numbered, with examples)

OUTPUT FORMAT (JSON):
{
    "rule_name": "Short descriptive name",
    "target_prompt": "qe" or "translation",
    "target_section": "Which section this belongs in (e.g., CORE DE→EN CALQUE DETECTION)",
    "rule_text": "The actual rule text to add to the prompt",
    "severity": "minor|major",
    "examples": [
        {"before": "...", "after": "...", "explanation": "..."}
    ],
    "rationale": "Why this rule is needed based on the data"
}"""


def generate_rule_proposal(pattern: dict, api_key: str,
                           model: str = "claude-sonnet-4-6") -> dict:
    """
    Feed a recurring pattern cluster to Claude and get a proposed prompt rule.

    Args:
        pattern: dict from analyze_feedback()["recurring_patterns"][i]
        api_key: Anthropic API key
        model: model to use for generation

    Returns:
        Proposal dict with rule text, rationale, and metadata
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Format examples for the prompt
    examples_text = []
    for i, ex in enumerate(pattern["examples"][:10]):
        examples_text.append(
            f"Example {i+1}:\n"
            f"  German source: {ex['de']}\n"
            f"  Original MT: {ex['original']}\n"
            f"  Translator's edit: {ex['edited']}\n"
            f"  QE rated: {ex['qe_rating']} ({ex.get('qe_category', 'n/a')})\n"
            f"  Edit classification: {ex['details']}\n"
        )

    user_message = (
        f"RECURRING PATTERN: {pattern['type']}\n"
        f"Occurrences: {pattern['count']}\n"
        f"QE catch rate: {pattern['catch_rate']}% "
        f"(caught {pattern['qe_caught_count']}, missed {pattern['qe_missed_count']})\n\n"
        f"TRANSLATOR EDITS:\n{''.join(examples_text)}\n\n"
        f"Based on these examples, draft a prompt rule that would help the QE model "
        f"detect this pattern. If this pattern is already partially covered by an "
        f"existing rule, suggest a refinement rather than a brand new rule."
    )

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.3,
        system=RULE_PROPOSAL_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = message.content[0].text.strip()

    # Parse the JSON response
    try:
        # Extract JSON from possible markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        proposal = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        brace_start = raw.find('{')
        brace_end = raw.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                proposal = json.loads(raw[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                proposal = {
                    "rule_name": f"auto_{pattern['type']}",
                    "target_prompt": "qe",
                    "target_section": "UNKNOWN",
                    "rule_text": raw,
                    "severity": "minor",
                    "examples": [],
                    "rationale": "Auto-generated — failed to parse structured response",
                }

    # Add metadata
    proposal["_metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pattern_type": pattern["type"],
        "pattern_count": pattern["count"],
        "catch_rate": pattern["catch_rate"],
        "model_used": model,
        "status": "pending",  # pending → approved → applied | rejected
        "ab_eval_result": None,
    }

    return proposal


def save_proposal(proposal: dict, proposals_dir: str = None) -> str:
    """Save a proposal to disk. Returns the file path."""
    out_dir = proposals_dir or PROPOSALS_DIR
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pattern_type = proposal.get("_metadata", {}).get("pattern_type", "unknown")
    filename = f"proposal_{timestamp}_{pattern_type}.json"
    path = os.path.join(out_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)

    return path


def load_proposals(proposals_dir: str = None, status: str = None) -> list:
    """Load all proposals, optionally filtered by status."""
    out_dir = proposals_dir or PROPOSALS_DIR
    if not os.path.exists(out_dir):
        return []

    proposals = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(out_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                prop = json.load(f)
            prop["_filename"] = fname
            if status is None or prop.get("_metadata", {}).get("status") == status:
                proposals.append(prop)
        except (json.JSONDecodeError, OSError):
            continue

    return proposals


def update_proposal_status(filename: str, new_status: str,
                           ab_result: dict = None, proposals_dir: str = None):
    """Update a proposal's status (approved/rejected/applied)."""
    out_dir = proposals_dir or PROPOSALS_DIR
    path = os.path.join(out_dir, filename)

    with open(path, "r", encoding="utf-8") as f:
        prop = json.load(f)

    prop["_metadata"]["status"] = new_status
    if ab_result:
        prop["_metadata"]["ab_eval_result"] = ab_result
    prop["_metadata"]["status_updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(prop, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Full pipeline: analyze → propose
# ---------------------------------------------------------------------------

def run_evolution_pipeline(api_key: str, model: str = "claude-sonnet-4-6",
                           min_pattern_count: int = 3,
                           min_miss_rate: float = 40.0) -> dict:
    """
    Run the full evolution pipeline:
    1. Analyze all feedback
    2. Find recurring patterns where QE misses > min_miss_rate%
    3. Generate rule proposals for each
    4. Save proposals for review

    Args:
        api_key: Anthropic API key
        model: model for proposal generation
        min_pattern_count: minimum occurrences to consider a pattern
        min_miss_rate: minimum QE miss rate (%) to trigger a proposal

    Returns:
        {
            "analysis": {...},
            "proposals_generated": 3,
            "proposal_files": ["proposal_20260327_001.json", ...],
        }
    """
    print("Stage 1: Analyzing feedback...", file=sys.stderr)
    analysis = analyze_feedback()

    print(f"  Total entries: {analysis['total_entries']}", file=sys.stderr)
    print(f"  Changed: {analysis['changed_entries']}", file=sys.stderr)
    print(f"  Unchanged: {analysis['unchanged_entries']}", file=sys.stderr)
    print(f"  QE false negatives: {analysis['qe_disagreements']['false_negative_count']}",
          file=sys.stderr)
    print(f"  QE false positives: {analysis['qe_disagreements']['false_positive_count']}",
          file=sys.stderr)

    # Filter to patterns worth proposing rules for
    actionable = []
    for pattern in analysis["recurring_patterns"]:
        if pattern["count"] < min_pattern_count:
            continue
        # Focus on patterns QE is missing
        miss_rate = 100 - pattern["catch_rate"]
        if miss_rate < min_miss_rate:
            continue
        actionable.append(pattern)

    print(f"\nStage 2: Found {len(actionable)} actionable patterns", file=sys.stderr)
    for p in actionable:
        print(f"  - {p['type']}: {p['count']} occurrences, "
              f"QE catches {p['catch_rate']}%", file=sys.stderr)

    # Generate proposals
    proposal_files = []
    if actionable and api_key:
        print(f"\nStage 3: Generating proposals...", file=sys.stderr)
        for pattern in actionable:
            try:
                print(f"  Generating proposal for: {pattern['type']}...", file=sys.stderr)
                proposal = generate_rule_proposal(pattern, api_key, model)
                path = save_proposal(proposal)
                proposal_files.append(os.path.basename(path))
                print(f"    Saved: {os.path.basename(path)}", file=sys.stderr)
            except Exception as e:
                print(f"    ERROR generating proposal for {pattern['type']}: {e}",
                      file=sys.stderr)

    return {
        "analysis": analysis,
        "proposals_generated": len(proposal_files),
        "proposal_files": proposal_files,
    }


# ---------------------------------------------------------------------------
# Report generation (for display in app or CLI)
# ---------------------------------------------------------------------------

def generate_feedback_report(analysis: dict = None) -> str:
    """Generate a human-readable report from feedback analysis."""
    if analysis is None:
        analysis = analyze_feedback()

    lines = []
    lines.append("=" * 60)
    lines.append("GLOSSWERK FEEDBACK ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Total confirmed segments: {analysis['total_entries']}")
    lines.append(f"  Accepted as-is: {analysis['unchanged_entries']}")
    lines.append(f"  Edited before confirming: {analysis['changed_entries']}")
    lines.append("")

    # QE accuracy
    disagree = analysis["qe_disagreements"]
    lines.append("QE ACCURACY:")
    lines.append(f"  False negatives (QE said good, translator edited): "
                 f"{disagree['false_negative_count']}")
    lines.append(f"  False positives (QE flagged, translator accepted as-is): "
                 f"{disagree['false_positive_count']}")
    lines.append("")

    # Edit type breakdown
    lines.append("EDIT TYPE BREAKDOWN:")
    for edit_type, count in sorted(analysis["edit_type_counts"].items(),
                                   key=lambda x: -x[1]):
        lines.append(f"  {edit_type}: {count}")
    lines.append("")

    # Recurring patterns
    if analysis["recurring_patterns"]:
        lines.append("RECURRING PATTERNS (potential prompt improvements):")
        lines.append("-" * 60)
        for p in analysis["recurring_patterns"]:
            status = "NEEDS RULE" if p["catch_rate"] < 60 else "PARTIALLY COVERED"
            lines.append(f"  {p['type']}: {p['count']} occurrences — "
                         f"QE catches {p['catch_rate']}% [{status}]")
            if p["examples"]:
                ex = p["examples"][0]
                lines.append(f"    Example: '{ex['original'][:80]}...'")
                lines.append(f"         →  '{ex['edited'][:80]}...'")
            lines.append("")

    # Pending proposals
    proposals = load_proposals(status="pending")
    if proposals:
        lines.append(f"PENDING PROPOSALS: {len(proposals)}")
        for prop in proposals:
            lines.append(f"  - {prop.get('rule_name', 'unnamed')}: "
                         f"{prop.get('rationale', '')[:100]}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GlossWerk Prompt Evolver")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze feedback and generate proposals")
    parser.add_argument("--report", action="store_true",
                        help="Print feedback analysis report (no proposals)")
    parser.add_argument("--list-proposals", action="store_true",
                        help="List all pending proposals")
    parser.add_argument("--approve", type=str,
                        help="Mark a proposal as approved (filename)")
    parser.add_argument("--reject", type=str,
                        help="Mark a proposal as rejected (filename)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--min-count", type=int, default=3,
                        help="Min occurrences to trigger proposal (default: 3)")
    parser.add_argument("--min-miss-rate", type=float, default=40.0,
                        help="Min QE miss rate %% to trigger proposal (default: 40)")
    args = parser.parse_args()

    if args.report:
        print(generate_feedback_report())
        return

    if args.list_proposals:
        proposals = load_proposals()
        if not proposals:
            print("No proposals found.")
            return
        for prop in proposals:
            meta = prop.get("_metadata", {})
            print(f"  [{meta.get('status', '?')}] {prop.get('_filename', '?')}")
            print(f"    Rule: {prop.get('rule_name', 'unnamed')}")
            print(f"    Pattern: {meta.get('pattern_type', '?')} "
                  f"(count={meta.get('pattern_count', '?')}, "
                  f"catch_rate={meta.get('catch_rate', '?')}%)")
            print(f"    Rationale: {prop.get('rationale', '')[:120]}")
            print()
        return

    if args.approve:
        update_proposal_status(args.approve, "approved")
        print(f"Approved: {args.approve}")
        return

    if args.reject:
        update_proposal_status(args.reject, "rejected")
        print(f"Rejected: {args.reject}")
        return

    if args.analyze:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        result = run_evolution_pipeline(
            api_key=api_key,
            model=args.model,
            min_pattern_count=args.min_count,
            min_miss_rate=args.min_miss_rate,
        )
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Proposals generated: {result['proposals_generated']}")
        for f in result['proposal_files']:
            print(f"  - {f}")
        if result['proposals_generated'] == 0:
            print("  (No patterns met the threshold for proposal generation)")
            if not api_key:
                print("  NOTE: No API key provided — skipped proposal generation")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
