"""
GlossWerk LLM Quality Estimator Module

Evaluates DE→EN patent translations using Claude as a QE system,
calibrated with few-shot examples from HTER-rated training data.

Inputs:
- German source sentences
- English translations (from translate.py)
- Structural risk scores (from analyze_structure.py)
- Few-shot calibration examples (from training_pairs.jsonl)

Outputs per segment:
- Rating: good / minor / major / critical
- Error category: terminology / reordering / omission / grammar / other
- Explanation: brief description of the issue
- Confidence: high / medium / low

The QE uses structural risk scores to calibrate skepticism:
high-risk sentences get extra scrutiny for reordering errors.

Usage:
    python quality_estimate.py --translations translations.json \
        --output qe_results.json --api-key YOUR_KEY

Requires: anthropic
Install:  pip install anthropic
"""

import json
import os
import re
import sys
import time
import random
from pathlib import Path

from prompt_layers import build_qe_prompt


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------

def load_training_pairs(jsonl_path, max_examples=500):
    """Load HTER-rated training pairs from JSONL file."""
    pairs = []
    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Deduplicate by segment_id + patent
                key = (entry.get("patent", ""), entry.get("segment_id", ""))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(entry)
            except json.JSONDecodeError:
                continue
            if len(pairs) >= max_examples:
                break
    return pairs


def select_few_shot_examples(training_pairs, n=30):
    """
    Select stratified few-shot examples for QE calibration.

    Strategy: pick examples across all rating levels so the LLM
    learns what "good" vs "minor" vs "major" vs "critical" actually
    looks like for patent translation.

    Target distribution:
    - good: 10 examples (so the LLM sees what publishable looks like)
    - minor: 10 examples (the tricky boundary — "close but needs a touch")
    - major: 7 examples (clear errors)
    - critical: 3 examples (if available)
    """
    by_rating = {"good": [], "minor": [], "major": [], "critical": []}
    for pair in training_pairs:
        rating = pair.get("rating", "").lower()
        if rating in by_rating:
            by_rating[rating].append(pair)

    targets = {"good": 10, "minor": 10, "major": 7, "critical": 3}
    selected = []

    for rating, target in targets.items():
        pool = by_rating.get(rating, [])
        if pool:
            k = min(target, len(pool))
            selected.extend(random.sample(pool, k))

    # If we don't have enough from some categories, fill from others
    remaining = n - len(selected)
    if remaining > 0:
        all_unused = [p for p in training_pairs if p not in selected]
        if all_unused:
            extra = min(remaining, len(all_unused))
            selected.extend(random.sample(all_unused, extra))

    return selected


def format_few_shot_examples(examples):
    """Format selected examples into the few-shot prompt section."""
    lines = ["Here are calibration examples showing how to rate patent translations:\n"]

    for i, ex in enumerate(examples):
        de = ex.get("de", "")
        mt = ex.get("deepl", ex.get("corrected", ""))
        rating = ex.get("rating", "unknown")
        notes = ex.get("notes", "")
        changed = ex.get("changed", False)

        lines.append(f"Example {i+1}:")
        lines.append(f"  German: {de[:200]}")
        lines.append(f"  Translation: {mt[:200]}")
        lines.append(f"  Rating: {rating}")
        if changed and notes:
            lines.append(f"  Note: {notes}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QE System Prompt
# ---------------------------------------------------------------------------

# QE_SYSTEM_PROMPT is now assembled from prompt_layers module.
# Kept as a module-level reference for backward compatibility.
QE_SYSTEM_PROMPT = None  # Use build_qe_system_prompt() instead


def build_qe_system_prompt(few_shot_examples=None, domain="patent"):
    """
    Build the full QE system prompt with optional few-shot examples.

    Delegates to prompt_layers.build_qe_prompt() for layered assembly:
      Tier 1 — Core DE→EN QE checks (domain-agnostic)
      Tier 2 — Domain overlay (patent, general, etc.)
      + Few-shot calibration examples

    Args:
        few_shot_examples: list of training pair dicts for calibration, or None
        domain: str — "patent", "general", etc.
    Returns:
        Complete QE system prompt string
    """
    few_shot_text = None
    if few_shot_examples:
        few_shot_text = format_few_shot_examples(few_shot_examples)

    return build_qe_prompt(domain=domain, few_shot_text=few_shot_text)


# ---------------------------------------------------------------------------
# QE API calls
# ---------------------------------------------------------------------------

def get_client(api_key):
    """Create Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def evaluate_translations(translations, api_key, model="claude-sonnet-4-6",
                          few_shot_examples=None, batch_size=20,
                          progress_callback=None, domain="patent"):
    """
    Evaluate translations using Claude as QE.

    Args:
        translations: list of dicts from translate.py output, each with:
            - source (German)
            - translation (English)
            - risk_level, risk_score, had_structural_hint
        api_key: Anthropic API key
        model: Claude model name
        few_shot_examples: list of training pair dicts for calibration
        batch_size: segments per API call
        progress_callback: fn(current, total)

    Returns:
        list of dicts: [{
            "index": 0,
            "rating": "good|minor|major|critical",
            "error_category": "terminology|reordering|omission|grammar|other",
            "explanation": "Brief description",
            "confidence": "high|medium|low",
            "risk_level": "low|medium|high",
            "risk_score": 0.0
        }, ...]
    """
    system_prompt = build_qe_system_prompt(few_shot_examples, domain=domain)
    n = len(translations)
    all_results = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = translations[batch_start:batch_end]

        batch_results = _evaluate_batch(batch, batch_start, api_key, model, system_prompt)
        all_results.extend(batch_results)

        if progress_callback:
            progress_callback(batch_end, n)

    return all_results


def _evaluate_batch(batch, global_offset, api_key, model, system_prompt):
    """Evaluate a batch of translation segments."""
    client = get_client(api_key)

    # Build evaluation input
    segments = []
    for i, entry in enumerate(batch):
        seg_num = i + 1
        risk = entry.get("risk_level", "unknown")
        score = entry.get("risk_score", 0.0)
        hint = ""
        if risk in ("medium", "high"):
            hint = f" [STRUCTURAL RISK: {risk}, score={score:.2f}]"

        segments.append(
            f"[{seg_num}]\n"
            f"  German: {entry.get('source', '')}\n"
            f"  English: {entry.get('translation', '')}{hint}"
        )

    segments_text = "\n\n".join(segments)

    user_message = (
        "Evaluate each numbered translation segment below. "
        "For each segment, return a JSON array with one object per segment.\n\n"
        "CRITICAL — FIELD ORDER: You MUST write 'analysis' FIRST, then 'rating'. "
        "This forces you to reason through the issue before committing to a severity level. "
        "If your analysis concludes the translation is actually acceptable, you MUST rate it 'good' — "
        "do not leave a 'minor' or 'major' rating that contradicts your own reasoning.\n\n"
        "```json\n"
        "[\n"
        '  {"index": 1, '
        '"analysis": "1-2 sentences MAX. Identify the specific error or say acceptable.", '
        '"rating": "good|minor|major|critical", '
        '"error_category": "terminology|reordering|omission|grammar|addition|other|none", '
        '"explanation": "Brief explanation or empty if good", '
        '"suggestion": "Improved translation if rating is not good, otherwise empty string", '
        '"confidence": "high|medium|low"},\n'
        "  ...\n"
        "]\n"
        "```\n\n"
        "KEEP ANALYSIS SHORT — 1-2 sentences maximum. Do not repeat the source or translation in the analysis.\n"
        "For segments rated 'good', use error_category 'none', empty explanation, empty suggestion, "
        "and analysis can be just 'Accurate and natural.'\n"
        "For non-good segments, ALWAYS provide a 'suggestion' field with your improved translation. "
        "This is critical — the suggestion must:\n"
        "  1. Be a COMPLETE, ready-to-use English sentence (not a fragment)\n"
        "  2. Fix ALL issues mentioned in your explanation, not just the first one\n"
        "  3. Wrap each changed/fixed portion in **double asterisks** for highlighting "
        "(e.g., 'The stent is inserted **via catheter** into the **patient\\'s artery**')\n"
        "  4. If the explanation mentions multiple errors, the suggestion must address every single one\n"
        "  5. Re-read your suggestion before finalizing — if it introduces lexical repetition "
        "(e.g., 'by means of a fastening means'), awkward word echoes, or reads worse than the original, "
        "rephrase to avoid the collision (e.g., use 'using a fastening means' or 'with a fastening means' instead)\n\n"
        "If you identify multiple DISTINCT errors in one segment (e.g., terminology + reordering, "
        "or omission + calque), list ALL errors in the explanation separated by semicolons, "
        "and escalate the rating per the MULTI-FAILURE ESCALATION rule.\n\n"
        "Be precise and consistent. Do not be overly generous — "
        "a translation with awkward word order IS a reordering problem even if meaning is preserved. "
        "But equally, do NOT flag standard patent English constructions as errors. "
        "'This object is achieved by' is standard patent phrasing, not a calque.\n\n"
        f"{segments_text}"
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        if not message.content:
            print(f"QE batch: empty response (stop_reason={message.stop_reason})", file=sys.stderr)
            return _fallback_results(batch, global_offset)

        raw = message.content[0].text.strip()
        stop_reason = getattr(message, "stop_reason", "unknown")

        # If truncated, try to repair the JSON before parsing
        if stop_reason == "max_tokens":
            print(f"QE batch: response truncated (max_tokens). Attempting JSON repair.", file=sys.stderr)
            raw = _repair_truncated_json(raw)

        parsed = _parse_qe_response(raw, len(batch))

        # Check if parsing actually worked — if we got all defaults, log the raw response
        if parsed and all(p.get("explanation", "").startswith("[QE FAILED") for p in parsed):
            print(f"QE batch: parse produced all-default results. stop_reason={stop_reason}", file=sys.stderr)
            print(f"QE batch: raw response (first 500 chars): {raw[:500]}", file=sys.stderr)

        # Attach risk info from input
        results = []
        for i, entry in enumerate(batch):
            qe = parsed[i] if i < len(parsed) else {
                "rating": "minor",
                "error_category": "other",
                "explanation": "[PARSE ERROR]",
                "suggestion": "",
                "confidence": "low"
            }
            results.append({
                "index": global_offset + i,
                "rating": qe.get("rating", "minor"),
                "error_category": qe.get("error_category", "other"),
                "explanation": qe.get("explanation", ""),
                "suggestion": qe.get("suggestion", ""),
                "confidence": qe.get("confidence", "medium"),
                "risk_level": entry.get("risk_level", "unknown"),
                "risk_score": entry.get("risk_score", 0.0),
            })
        return results

    except Exception as e:
        import traceback
        print(f"ERROR in QE batch (offset={global_offset}, size={len(batch)}): {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Retry with smaller batches before giving up
        if len(batch) > 3:
            print(f"Retrying with smaller batches (split {len(batch)} → {len(batch)//2} + {len(batch) - len(batch)//2})...", file=sys.stderr)
            mid = len(batch) // 2
            results_a = _evaluate_batch(batch[:mid], global_offset, api_key, model, system_prompt)
            results_b = _evaluate_batch(batch[mid:], global_offset + mid, api_key, model, system_prompt)
            return results_a + results_b
        return _fallback_results(batch, global_offset)


def _repair_truncated_json(raw_text):
    """Attempt to repair JSON that was truncated mid-stream (max_tokens hit)."""
    # Strip trailing incomplete tokens
    text = raw_text.rstrip()

    # If inside a code block, extract what we have
    code_match = re.search(r'```(?:json)?\s*\n?(.*)', text, re.DOTALL)
    if code_match:
        text = code_match.group(1).rstrip().rstrip('`').rstrip()

    # Find the last complete JSON object (ends with })
    last_brace = text.rfind('}')
    if last_brace >= 0:
        text = text[:last_brace + 1]

    # Ensure array is closed
    bracket_start = text.find('[')
    if bracket_start >= 0:
        # Remove any trailing comma after last }
        text = text.rstrip().rstrip(',').rstrip()
        if not text.endswith(']'):
            text += ']'

    return text


def _parse_qe_response(raw_text, expected_count):
    """Parse the JSON array from Claude's QE response.

    Handles: code blocks, truncated JSON, individual objects.
    """
    # Try to extract JSON from markdown code block
    code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
    if code_match:
        raw_text = code_match.group(1).strip()

    # Try to find JSON array
    bracket_start = raw_text.find('[')
    bracket_end = raw_text.rfind(']')
    if bracket_start >= 0 and bracket_end > bracket_start:
        json_str = raw_text[bracket_start:bracket_end + 1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) >= 1:
                return parsed
        except json.JSONDecodeError:
            # Try repairing: truncate to last complete object
            repaired = _repair_truncated_json(json_str)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, list) and len(parsed) >= 1:
                    print(f"QE parse: repaired truncated JSON, got {len(parsed)}/{expected_count} segments", file=sys.stderr)
                    return parsed
            except json.JSONDecodeError:
                pass

    # Try raw parse
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract individual JSON objects using a more robust approach
    # Match balanced braces (handles nested quotes with escaped chars)
    objects = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(raw_text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(raw_text[start:i + 1])
                    if "rating" in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1

    if objects:
        print(f"QE parse: extracted {len(objects)}/{expected_count} individual objects", file=sys.stderr)
        return objects

    # Last resort: return defaults
    print(f"QE parse: FAILED all strategies. Raw length={len(raw_text)}, first 300 chars: {raw_text[:300]}", file=sys.stderr)
    return [{"rating": "minor", "error_category": "other",
             "explanation": "[QE FAILED — review manually]",
             "suggestion": "",
             "confidence": "low"}] * expected_count


def _fallback_results(batch, global_offset):
    """Generate fallback results when API call fails."""
    results = []
    for i, entry in enumerate(batch):
        results.append({
            "index": global_offset + i,
            "rating": "minor",
            "error_category": "other",
            "explanation": "[QE FAILED — review manually]",
            "suggestion": "",
            "confidence": "low",
            "risk_level": entry.get("risk_level", "unknown"),
            "risk_score": entry.get("risk_score", 0.0),
        })
    return results


# ---------------------------------------------------------------------------
# Triage summary
# ---------------------------------------------------------------------------

def compute_triage(qe_results):
    """
    Compute triage buckets from QE results.

    Returns:
        dict with:
        - green: list of indices (publishable)
        - orange: list of indices (quick review)
        - red: list of indices (full edit)
        - summary: aggregate stats
    """
    green = []   # good rating, high/medium confidence
    orange = []  # minor rating, or good with low confidence
    red = []     # major or critical

    for r in qe_results:
        rating = r.get("rating", "minor")
        confidence = r.get("confidence", "medium")

        if rating == "good" and confidence in ("high", "medium"):
            green.append(r["index"])
        elif rating == "good" and confidence == "low":
            orange.append(r["index"])
        elif rating == "minor":
            orange.append(r["index"])
        else:  # major or critical
            red.append(r["index"])

    total = len(qe_results)
    summary = {
        "total_segments": total,
        "green_count": len(green),
        "orange_count": len(orange),
        "red_count": len(red),
        "green_pct": round(len(green) / total * 100, 1) if total else 0,
        "orange_pct": round(len(orange) / total * 100, 1) if total else 0,
        "red_pct": round(len(red) / total * 100, 1) if total else 0,
    }

    # Error type breakdown
    from collections import Counter
    error_types = Counter(r.get("error_category", "other")
                          for r in qe_results if r.get("rating") != "good")
    summary["error_breakdown"] = dict(error_types)

    # Cross-reference: how many high-risk sentences ended up green?
    high_risk_green = sum(1 for r in qe_results
                         if r["index"] in green and r.get("risk_level") == "high")
    high_risk_total = sum(1 for r in qe_results if r.get("risk_level") == "high")
    summary["high_risk_green"] = high_risk_green
    summary["high_risk_total"] = high_risk_total

    return {
        "green": green,
        "orange": orange,
        "red": red,
        "summary": summary,
    }


def print_triage_summary(triage, qe_results):
    """Print human-readable triage summary."""
    s = triage["summary"]
    print(f"\n{'='*60}")
    print(f"QE TRIAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments: {s['total_segments']}")
    print()
    print(f"  GREEN  (publishable):  {s['green_count']:3d}  ({s['green_pct']:.1f}%)")
    print(f"  ORANGE (quick review): {s['orange_count']:3d}  ({s['orange_pct']:.1f}%)")
    print(f"  RED    (full edit):    {s['red_count']:3d}  ({s['red_pct']:.1f}%)")
    print()

    if s.get("error_breakdown"):
        print(f"Error type breakdown (non-good segments):")
        for etype, count in sorted(s["error_breakdown"].items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count}")
        print()

    if s["high_risk_total"] > 0:
        print(f"High structural risk: {s['high_risk_total']} sentences")
        print(f"  Of those, {s['high_risk_green']} ended up GREEN "
              f"(structural hints helped translation)")
        print()

    # Show top RED segments
    red_segments = [r for r in qe_results if r["index"] in triage["red"]]
    if red_segments:
        print(f"RED SEGMENTS (need full edit):")
        print(f"{'-'*60}")
        for r in red_segments[:10]:
            print(f"  [{r['index']}] {r['rating']} — {r['error_category']}: {r['explanation'][:100]}")
        if len(red_segments) > 10:
            print(f"  ... and {len(red_segments) - 10} more")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="QE evaluation of patent translations")
    parser.add_argument("--translations", required=True,
                        help="JSON file from translate.py")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--training-pairs", default=None,
                        help="JSONL file for few-shot calibration")
    parser.add_argument("--n-examples", type=int, default=30,
                        help="Number of few-shot examples (default: 30)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Segments per QE API call (default: 20)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set ANTHROPIC_API_KEY")
        sys.exit(1)

    # Load translations
    with open(args.translations, "r", encoding="utf-8") as f:
        trans_data = json.load(f)
    translations = trans_data.get("translations", [])
    print(f"Loaded {len(translations)} translated segments")

    # Load few-shot examples
    few_shot = None
    if args.training_pairs:
        tp_path = Path(args.training_pairs)
        if tp_path.exists():
            all_pairs = load_training_pairs(str(tp_path))
            few_shot = select_few_shot_examples(all_pairs, args.n_examples)
            print(f"Selected {len(few_shot)} few-shot calibration examples")
        else:
            print(f"WARNING: Training pairs not found: {tp_path}")

    # Run QE
    def progress(current, total):
        print(f"  Evaluated {current}/{total} segments...")

    qe_results = evaluate_translations(
        translations=translations,
        api_key=api_key,
        model=args.model,
        few_shot_examples=few_shot,
        batch_size=args.batch_size,
        progress_callback=progress,
    )

    # Compute triage
    triage = compute_triage(qe_results)
    print_triage_summary(triage, qe_results)

    # Save output
    output = {
        "metadata": {
            "source": trans_data.get("metadata", {}),
            "qe_model": args.model,
            "n_few_shot": len(few_shot) if few_shot else 0,
        },
        "qe_results": qe_results,
        "triage": triage,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
