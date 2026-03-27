"""
GlossWerk Feedback Logger

Logs confirmed/edited translation segments as training data for QE calibration.
Every time a user confirms a segment (with or without edits), we capture:
- The German source
- The original MT output
- The final confirmed version
- Whether it was changed
- The QE rating that was assigned

This data feeds back into training_pairs.jsonl for few-shot QE calibration,
creating a continuous improvement loop.

Usage:
    from feedback_logger import log_confirmed_segment, export_session_feedback
"""

import json
import os
from datetime import datetime, timezone


# Default path for feedback data — uses DATA_DIR env var if set (for Railway persistent volumes)
_data_root = os.environ.get("DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
FEEDBACK_DIR = _data_root
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback_pairs.jsonl")
TRAINING_FILE = os.path.join(_data_root, "hter_training", "training_pairs.jsonl")


def log_confirmed_segment(source_de, original_translation, confirmed_translation,
                          qe_rating, qe_category, segment_index,
                          doc_name="unknown", client_id="unknown",
                          feedback_path=None):
    """
    Log a single confirmed segment to the feedback JSONL file.

    Args:
        source_de: German source sentence
        original_translation: The MT output before any edits
        confirmed_translation: The final text the user confirmed
        qe_rating: The QE rating assigned (good/minor/major/critical)
        qe_category: The QE error category
        segment_index: Segment number in the document
        doc_name: Name of the source document
        client_id: Client identifier (company name or demo code)
        feedback_path: Override path for the feedback file
    """
    path = feedback_path or FEEDBACK_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)

    changed = original_translation.strip() != confirmed_translation.strip()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patent": doc_name,
        "segment_id": segment_index,
        "client_id": client_id,
        "de": source_de,
        "deepl": original_translation,  # "deepl" for backward compat with training format
        "corrected": confirmed_translation,
        "rating": _derive_rating(qe_rating, changed),
        "changed": changed,
        "notes": f"QE rated: {qe_rating} ({qe_category})" if qe_category else "",
        "qe_rating_original": qe_rating,
        "qe_category_original": qe_category,
        "source": "user_feedback",
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def _derive_rating(qe_rating, changed):
    """
    Derive the training rating from QE rating + user action.

    Logic:
    - User confirmed without edits → trust the translation is at least "good"
      (even if QE said minor/major — the human overrides the machine)
    - User edited and confirmed → "minor" at minimum
      (the translation needed work, but the user fixed it)
    - QE said critical and user edited → keep "major"
      (was bad enough to need significant changes)
    """
    if not changed:
        return "good"
    elif qe_rating in ("critical",):
        return "major"
    elif qe_rating in ("major",):
        return "minor"  # User fixed it, so it's now training data for "what minor looks like"
    else:
        return "minor"


def log_session_feedback(translations, confirmed_dict, qe_results, doc_name="unknown"):
    """
    Bulk-log all confirmed segments from a session.
    Called at export time to capture everything at once.

    Args:
        translations: list of translation dicts (with 'source', 'translation', 'index')
        confirmed_dict: dict {index: confirmed_text}
        qe_results: list of QE result dicts
        doc_name: source document name

    Returns:
        int: number of segments logged
    """
    qe_by_idx = {r["index"]: r for r in qe_results}
    count = 0

    for trans in translations:
        idx = trans["index"]
        if idx in confirmed_dict:
            qe = qe_by_idx.get(idx, {})
            log_confirmed_segment(
                source_de=trans.get("source", ""),
                original_translation=trans.get("translation", ""),
                confirmed_translation=confirmed_dict[idx],
                qe_rating=qe.get("rating", "unknown"),
                qe_category=qe.get("error_category", ""),
                segment_index=idx,
                doc_name=doc_name,
            )
            count += 1

    return count


def merge_feedback_to_training(feedback_path=None, training_path=None, min_entries=20):
    """
    Merge accumulated feedback into the main training_pairs.jsonl.
    Only merges when there are at least min_entries new feedback entries.

    Args:
        feedback_path: path to feedback_pairs.jsonl
        training_path: path to training_pairs.jsonl
        min_entries: minimum feedback entries before merging

    Returns:
        int: number of entries merged, or 0 if threshold not met
    """
    fb_path = feedback_path or FEEDBACK_FILE
    tr_path = training_path or TRAINING_FILE

    if not os.path.exists(fb_path):
        return 0

    # Count feedback entries
    with open(fb_path, "r", encoding="utf-8") as f:
        entries = [line.strip() for line in f if line.strip()]

    if len(entries) < min_entries:
        return 0

    # Append to training file
    os.makedirs(os.path.dirname(tr_path), exist_ok=True)
    with open(tr_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")

    # Clear feedback file (it's been merged)
    with open(fb_path, "w", encoding="utf-8") as f:
        pass  # empty the file

    return len(entries)


def get_feedback_stats(feedback_path=None):
    """Get summary stats of accumulated feedback."""
    path = feedback_path or FEEDBACK_FILE
    if not os.path.exists(path):
        return {"total": 0, "changed": 0, "unchanged": 0, "by_qe_rating": {}}

    stats = {"total": 0, "changed": 0, "unchanged": 0, "by_qe_rating": {}}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                stats["total"] += 1
                if entry.get("changed"):
                    stats["changed"] += 1
                else:
                    stats["unchanged"] += 1
                qr = entry.get("qe_rating_original", "unknown")
                stats["by_qe_rating"][qr] = stats["by_qe_rating"].get(qr, 0) + 1
            except json.JSONDecodeError:
                continue

    return stats
