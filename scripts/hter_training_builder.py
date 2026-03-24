"""
GlossWerk HTER Evaluation & Training Data Builder

Upload a DE patent .docx and its EN reference .docx.
The app:
  1. Sentence-splits both documents
  2. Translates the FULL German document in one Claude API call
     (maintaining terminology consistency across the entire patent)
  3. Shows you Claude MT vs Human Reference side by side
  4. You rate each segment and make corrections
  5. Every correction becomes a Claude MT → corrected pair for LoRA training
  6. Builds a glossary from your term corrections — glossary terms are
     injected into the system prompt for subsequent patents
  7. Tags each correction as terminology, syntax/reordering, or both
     for error analysis

Run: streamlit run hter_training_builder.py

Requires: ANTHROPIC_API_KEY environment variable (or enter in app)
Install: pip install anthropic streamlit python-docx
"""

import streamlit as st
import json
import os
import re
import time
from datetime import datetime

# --- Claude translation config ---
DEFAULT_SYSTEM_PROMPT = """You are an expert DE→EN patent translator with deep knowledge of European patent conventions.

Rules:
- Maintain precise technical terminology CONSISTENTLY throughout the entire document
- If you translate a German term a certain way in sentence 1, use the SAME English term every subsequent time
- Use "FIG." (not "Fig.") for figure references, per US patent convention
- Preserve claim structure and legal phrasing exactly
- Keep numbered paragraph references (e.g., [0012], [0021]) EXACTLY as they appear in the source — do NOT remove, reformat, or omit them
- Translate noun compounds precisely — do not simplify or paraphrase
- Produce exactly ONE English sentence per German input sentence — do NOT split a German sentence into multiple English sentences, even if it is long
- If a German term has a standard patent translation, use it consistently

Information structure and word order:
- German builds toward informationally heavy material at the end of the sentence (end-weight, end-focus). English front-loads key information.
- When translating long German sentences, RESTRUCTURE the English to place key information earlier rather than preserving German constituent order.
- Do not mirror German clause order in English. Reorder so the English reads naturally, with important content appearing earlier in the sentence.
- Pay special attention to sentences with subordinate clauses: the finite verb at the end of German subordinate clauses often means the most important information arrives last. In English, bring that information forward."""

AVAILABLE_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]


# --- Sentence splitting ---
def split_sentences(text):
    if not text or not text.strip():
        return []
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # Protect common abbreviations from splitting
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
        'e.g.': 'EG__', 'i.e.': 'IE__',
        'et al.': 'ETAL__', 'approx.': 'APPROX__',
        'No.': 'NO__', 'Nos.': 'NOS__',
        'vs.': 'VS__', 'Dr.': 'DR__',
        'St.': 'ST__', 'Corp.': 'CORP__',
        'Inc.': 'INC__', 'Ltd.': 'LTD__',
        'U.S.': 'US__', 'S.': 'S__',
    }

    for orig, placeholder in protected.items():
        text = text.replace(orig, placeholder)

    # Split on period/question/exclamation followed by space and capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ\[(])', text)

    # Restore abbreviations
    sentences = []
    for part in parts:
        for orig, placeholder in protected.items():
            part = part.replace(placeholder, orig)
        part = part.strip()
        if len(part) > 15:
            sentences.append(part)
    return sentences


def extract_text_from_docx(filepath):
    from docx import Document as DocxDocument
    doc = DocxDocument(filepath)
    full_text = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text and len(text) > 10:
            full_text.append(text)
    return "\n".join(full_text)


# --- Claude translation functions ---
def get_claude_client(api_key):
    """Create the Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def build_system_prompt(base_prompt, glossary):
    """Append glossary terms to the system prompt if any exist."""
    if not glossary:
        return base_prompt

    glossary_lines = "\n".join(f"- {de} → {en}" for de, en in glossary.items())
    return (
        f"{base_prompt}\n\n"
        f"MANDATORY TERMINOLOGY — always use these translations:\n"
        f"{glossary_lines}"
    )


def translate_full_document(sentences, api_key, model, system_prompt):
    """Translate the entire document in ONE API call for terminology consistency.

    Sends all German sentences as a numbered list. Claude returns a numbered
    list of English translations. This gives Claude full document context so
    it can maintain consistent terminology across all segments.
    """
    client = get_claude_client(api_key)

    # Build numbered input
    numbered_input = "\n".join(
        f"[{i+1}] {sent}" for i, sent in enumerate(sentences)
    )

    user_message = (
        f"Translate each numbered German patent sentence below into English. "
        f"Return ONLY the translations as a numbered list in the same format: "
        f"[1] translation, [2] translation, etc. "
        f"One English sentence per German sentence. "
        f"Preserve all paragraph reference numbers like [0012] within the text. "
        f"Do not add commentary, notes, or explanations.\n\n"
        f"{numbered_input}"
    )

    message = client.messages.create(
        model=model,
        max_tokens=16384,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    if not message.content or len(message.content) == 0:
        return ["[EMPTY RESPONSE — translation failed]"] * len(sentences)

    raw_output = message.content[0].text.strip()

    # Parse numbered output back into list
    translations = parse_numbered_output(raw_output, len(sentences))

    return translations


def parse_numbered_output(raw_text, expected_count):
    """Parse Claude's numbered translation output back into a list.

    Handles formats like:
      [1] Translation one.
      [2] Translation two.
    or:
      1. Translation one.
      2. Translation two.
    """
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


def translate_claude_fallback(sentences, api_key, model, system_prompt,
                              progress_bar=None):
    """Fallback: translate sentence-by-sentence if full-document call fails."""
    client = get_claude_client(api_key)
    translations = []

    for i, sent in enumerate(sentences):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user",
                     "content": f"Translate this German patent text to English:\n\n{sent}"}
                ],
            )
            if message.content and len(message.content) > 0:
                translations.append(message.content[0].text.strip())
            else:
                translations.append(
                    f"[EMPTY RESPONSE — review manually: {sent[:50]}]"
                )
        except Exception as e:
            translations.append(f"[TRANSLATION ERROR: {e}]")

        if progress_bar is not None:
            progress_bar.progress(
                (i + 1) / len(sentences),
                text=f"Translating segment {i+1}/{len(sentences)} (fallback mode)"
            )

    return translations


# --- Data persistence ---
DATA_DIR = r"C:\glosswerk\data\hter_training"
PAIRS_FILE = os.path.join(DATA_DIR, "training_pairs.jsonl")
GLOSSARY_FILE = os.path.join(DATA_DIR, "glossary.tsv")
STATE_FILE = os.path.join(DATA_DIR, "session_state.json")

os.makedirs(DATA_DIR, exist_ok=True)


def load_existing_pairs():
    pairs = []
    if os.path.exists(PAIRS_FILE):
        with open(PAIRS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    return pairs


def save_pair(pair):
    with open(PAIRS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def load_glossary():
    glossary = {}
    if os.path.exists(GLOSSARY_FILE):
        with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    glossary[parts[0]] = parts[1]
    return glossary


def save_glossary_term(de_term, en_term):
    with open(GLOSSARY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{de_term}\t{en_term}\n")


def save_session_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def load_session_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# --- Navigation callbacks ---
# These run BEFORE the rerun, so session_state is updated before
# the slider reads it.
def go_previous():
    if st.session_state.get("current_idx", 0) > 0:
        st.session_state["current_idx"] -= 1


def go_next():
    total = len(st.session_state.get("de_sents", []))
    if st.session_state.get("current_idx", 0) < total - 1:
        st.session_state["current_idx"] += 1


# --- Streamlit App ---
def main():
    st.set_page_config(page_title="GlossWerk HTER Builder", layout="wide")
    st.title("GlossWerk — HTER Evaluation & Training Data Builder")

    # Sidebar stats
    existing_pairs = load_existing_pairs()
    glossary = load_glossary()

    st.sidebar.header("Progress")
    st.sidebar.metric("Training pairs collected", len(existing_pairs))
    st.sidebar.metric("Glossary terms", len(glossary))

    ratings = {"good": 0, "minor": 0, "major": 0, "critical": 0}
    error_types = {"terminology": 0, "syntax": 0, "both": 0}
    for p in existing_pairs:
        r = p.get("rating", "")
        if r in ratings:
            ratings[r] += 1
        et = p.get("error_type", "")
        if et in error_types:
            error_types[et] += 1

    st.sidebar.write("**Ratings breakdown:**")
    for r, c in ratings.items():
        if c > 0:
            st.sidebar.write(f"  {r}: {c}")

    if any(v > 0 for v in error_types.values()):
        st.sidebar.write("**Error types:**")
        for et, c in error_types.items():
            if c > 0:
                st.sidebar.write(f"  {et}: {c}")

    st.sidebar.divider()

    # --- Model config in sidebar ---
    st.sidebar.header("Claude API Settings")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("Anthropic API Key:", type="password")
    else:
        st.sidebar.success("API key loaded from ANTHROPIC_API_KEY")

    model = st.sidebar.selectbox("Model", AVAILABLE_MODELS, index=0)

    with st.sidebar.expander("System prompt (editable)", expanded=False):
        system_prompt = st.text_area(
            "Patent translation prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=300,
            key="system_prompt",
        )

    # Show glossary status
    if glossary:
        st.sidebar.success(
            f"Glossary active: {len(glossary)} terms injected into prompt"
        )
        with st.sidebar.expander("View glossary terms"):
            for de, en in sorted(glossary.items()):
                st.sidebar.write(f"  {de} → {en}")

    st.sidebar.divider()
    st.sidebar.header("Export")
    if st.sidebar.button("Export training DB"):
        export_to_db(existing_pairs)
        st.sidebar.success("Exported to training DB!")

    if not api_key:
        st.warning("Set ANTHROPIC_API_KEY env var or enter your key in the sidebar.")
        return

    # Build final system prompt with glossary
    full_system_prompt = build_system_prompt(system_prompt, glossary)

    # File upload
    st.header("1. Upload Patent Documents")
    col1, col2 = st.columns(2)

    with col1:
        de_file = st.file_uploader(
            "German patent (.docx)", type=["docx"], key="de_upload"
        )
    with col2:
        en_file = st.file_uploader(
            "English reference (.docx)", type=["docx"], key="en_upload"
        )

    # Or paste text directly
    st.header("Or paste text directly")
    col1, col2 = st.columns(2)
    with col1:
        de_text = st.text_area("German text", height=200, key="de_text")
    with col2:
        en_text = st.text_area("English reference text", height=200, key="en_text")

    patent_name = st.text_input("Patent name/number (for tracking)", value="")

    if st.button("Process", type="primary"):
        # Get text from files or text areas
        if de_file:
            temp_de = os.path.join(DATA_DIR, "temp_de.docx")
            with open(temp_de, "wb") as f:
                f.write(de_file.read())
            de_text = extract_text_from_docx(temp_de)

        if en_file:
            temp_en = os.path.join(DATA_DIR, "temp_en.docx")
            with open(temp_en, "wb") as f:
                f.write(en_file.read())
            en_text = extract_text_from_docx(temp_en)

        if not de_text:
            st.error("No German text provided!")
            return

        # Split sentences
        de_sents = split_sentences(de_text)
        en_sents = split_sentences(en_text) if en_text else []

        st.info(
            f"German sentences: {len(de_sents)} | "
            f"English reference sentences: {len(en_sents)}"
        )

        # Translate entire document in one call
        with st.spinner(
            f"Translating full document with {model} "
            f"({len(de_sents)} segments in one call for terminology consistency)..."
        ):
            try:
                mt_sents = translate_full_document(
                    de_sents, api_key, model, full_system_prompt
                )

                # Check for excessive parse errors
                error_count = sum(
                    1 for s in mt_sents if "[PARSE ERROR" in s or "[EMPTY" in s
                )
                if error_count > len(de_sents) * 0.3:
                    st.warning(
                        f"Full-document translation had {error_count} parse errors. "
                        f"Falling back to sentence-by-sentence..."
                    )
                    progress_bar = st.progress(0, text="Falling back...")
                    mt_sents = translate_claude_fallback(
                        de_sents, api_key, model, full_system_prompt, progress_bar
                    )
                    progress_bar.progress(1.0, text="Translation complete!")

            except Exception as e:
                st.warning(
                    f"Full-document call failed ({e}). "
                    f"Falling back to sentence-by-sentence..."
                )
                progress_bar = st.progress(0, text="Falling back...")
                mt_sents = translate_claude_fallback(
                    de_sents, api_key, model, full_system_prompt, progress_bar
                )
                progress_bar.progress(1.0, text="Translation complete!")

        st.success("Translation complete!")

        # Store in session state
        st.session_state["de_sents"] = de_sents
        st.session_state["en_sents"] = en_sents
        st.session_state["mt_sents"] = mt_sents
        st.session_state["mt_source"] = f"claude ({model})"
        st.session_state["patent_name"] = patent_name
        st.session_state["current_idx"] = 0

        # Save state
        save_session_state({
            "de_sents": de_sents,
            "en_sents": en_sents,
            "mt_sents": mt_sents,
            "mt_source": f"claude ({model})",
            "patent_name": patent_name,
            "current_idx": 0,
        })

    # --- Evaluation interface ---
    if "de_sents" not in st.session_state:
        saved = load_session_state()
        if saved.get("de_sents"):
            st.session_state.update(saved)

    if "de_sents" in st.session_state and st.session_state["de_sents"]:
        st.divider()
        st.header("2. Evaluate Segments")

        de_sents = st.session_state["de_sents"]
        en_sents = st.session_state.get("en_sents", [])
        mt_sents = st.session_state.get(
            "mt_sents", st.session_state.get("deepl_sents", [])
        )
        mt_source = st.session_state.get("mt_source", "unknown")
        patent_name = st.session_state.get("patent_name", "")

        total = len(de_sents)

        st.caption(f"MT source: **{mt_source}**")

        # Navigation — use on_click callbacks so state updates before rerun
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("← Previous", on_click=go_previous, key="btn_prev")
        with col3:
            st.button("Next →", on_click=go_next, key="btn_next")

        idx = st.session_state.get("current_idx", 0)
        # Clamp in case state is stale
        idx = max(0, min(idx, total - 1))

        with col2:
            new_idx = st.slider("Segment", 0, total - 1, idx, key="seg_slider")
            if new_idx != idx:
                st.session_state["current_idx"] = new_idx
                idx = new_idx

        st.subheader(f"Segment {idx + 1} / {total}")

        # Keyboard shortcut hint
        st.caption("Tip: use ← Previous / Next → buttons or the slider to navigate")

        # Show source
        st.markdown("**German source:**")
        st.text_area(
            "DE", de_sents[idx], height=80, disabled=True, key=f"de_{idx}"
        )

        # Show MT and reference side by side
        col1, col2 = st.columns(2)

        with col1:
            mt_label = (
                "Claude MT" if "claude" in mt_source.lower() else "MT output"
            )
            st.markdown(f"**{mt_label}:**")
            st.text_area(
                mt_label,
                mt_sents[idx] if idx < len(mt_sents) else "",
                height=100,
                disabled=True,
                key=f"mt_{idx}",
            )

        with col2:
            ref = (
                en_sents[idx]
                if idx < len(en_sents)
                else "(no reference available)"
            )
            st.markdown("**Human reference:**")
            st.text_area(
                "Reference", ref, height=100, disabled=True, key=f"ref_{idx}"
            )

        # Rating
        st.markdown("---")
        st.markdown("**Your evaluation:**")

        rating = st.radio(
            "MT quality for this segment:",
            [
                "good (no changes needed)",
                "minor (1-2 small fixes)",
                "major (significant corrections)",
                "critical (needs retranslation)",
            ],
            horizontal=True,
            key=f"rating_{idx}",
        )

        # Error type tag
        error_type = st.radio(
            "Primary error type (if not good):",
            [
                "n/a (segment is good)",
                "terminology (wrong term, inconsistent term)",
                "syntax (word order, reordering, clause structure)",
                "both (terminology + syntax issues)",
            ],
            horizontal=True,
            key=f"error_type_{idx}",
        )

        # Correction
        st.markdown("**Your corrected translation** (edit the MT output):")
        default_correction = mt_sents[idx] if idx < len(mt_sents) else ""
        corrected = st.text_area(
            "Correction",
            value=default_correction,
            height=100,
            key=f"corrected_{idx}",
        )

        # Terminology capture
        st.markdown("**Capture terminology** (optional):")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            de_term = st.text_input("German term", key=f"de_term_{idx}")
        with col2:
            en_term = st.text_input("English term", key=f"en_term_{idx}")
        with col3:
            if st.button("Add term", key=f"add_term_{idx}"):
                if de_term and en_term:
                    save_glossary_term(de_term, en_term)
                    st.success(f"Added: {de_term} → {en_term}")

        # Notes
        notes = st.text_input("Notes (optional)", key=f"notes_{idx}")

        # Save
        if st.button("Save & Next", type="primary", key=f"save_{idx}"):
            rating_short = rating.split(" ")[0]
            error_type_short = error_type.split(" ")[0]
            if error_type_short == "n/a":
                error_type_short = ""
            changed = corrected.strip() != default_correction.strip()

            pair = {
                "timestamp": datetime.now().isoformat(),
                "patent": patent_name,
                "segment_id": idx + 1,
                "de": de_sents[idx],
                "mt": mt_sents[idx] if idx < len(mt_sents) else "",
                "mt_source": mt_source,
                "reference": en_sents[idx] if idx < len(en_sents) else "",
                "corrected": corrected,
                "rating": rating_short,
                "error_type": error_type_short,
                "changed": changed,
                "notes": notes,
            }

            save_pair(pair)
            st.success(f"Saved segment {idx + 1} ({rating_short})")

            # Auto advance
            if idx < total - 1:
                st.session_state["current_idx"] = idx + 1
                st.rerun()


def export_to_db(pairs):
    """Export collected pairs to SQLite for LoRA training."""
    import sqlite3

    db_path = os.path.join(DATA_DIR, "hter_training.db")
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src TEXT NOT NULL,
            ref TEXT NOT NULL,
            mt_claude TEXT,
            mt_deepl TEXT,
            mt_opus TEXT,
            mt_source TEXT,
            error_type TEXT,
            ipc_code TEXT DEFAULT 'A61F_hter',
            publication TEXT,
            section TEXT DEFAULT 'description',
            split TEXT DEFAULT 'train'
        )
    """)

    # Only export pairs where corrections were made
    corrected_pairs = [p for p in pairs if p.get("changed", False)]

    import random
    random.seed(42)
    random.shuffle(corrected_pairs)

    n = len(corrected_pairs)
    n_val = max(1, n // 10)

    conn.execute("DELETE FROM domain_pairs WHERE ipc_code='A61F_hter'")

    for i, p in enumerate(corrected_pairs):
        split = "val" if i < n_val else "train"
        mt_source = p.get("mt_source", "unknown")
        mt_text = p.get("mt", p.get("deepl", ""))

        mt_claude = mt_text if "claude" in mt_source.lower() else None
        mt_deepl = mt_text if "deepl" in mt_source.lower() else p.get("deepl")

        conn.execute(
            """INSERT INTO domain_pairs
               (src, ref, mt_claude, mt_deepl, mt_source, error_type,
                ipc_code, publication, split)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                p["de"],
                p["corrected"],
                mt_claude,
                mt_deepl,
                mt_source,
                p.get("error_type", ""),
                "A61F_hter",
                p.get("patent", ""),
                split,
            ),
        )

    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM domain_pairs").fetchone()[0]
    st.sidebar.write(f"Exported {total} corrected pairs to {db_path}")

    conn.close()


if __name__ == "__main__":
    main()
