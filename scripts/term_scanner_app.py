"""
GlossWerk Terminology Scanner — Demo App

Upload a German patent .docx → extract nouns, technical adjectives, patent verbs →
get English translation proposals via Claude → translator reviews and selects →
export glossary TSV for translation pipeline.

Run: streamlit run term_scanner_app.py
Requires: ANTHROPIC_API_KEY environment variable
Install: pip install anthropic streamlit python-docx
"""

import json
import os
import sys
import tempfile

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import importlib.util

# Import from hyphenated directory name
_spec = importlib.util.spec_from_file_location(
    "extract_terms",
    os.path.join(PROJECT_ROOT, "skills", "glosswerk-term-scanner", "scripts", "extract_terms.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_text_from_docx = _mod.extract_text_from_docx
extract_sentences = _mod.extract_sentences
extract_nouns_heuristic = _mod.extract_nouns_heuristic
extract_nouns_spacy = _mod.extract_nouns_spacy
extract_technical_adjectives = _mod.extract_technical_adjectives
extract_patent_verbs = _mod.extract_patent_verbs
cluster_variants = _mod.cluster_variants
HAS_SPACY = _mod.HAS_SPACY

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# --- Page config ---
st.set_page_config(
    page_title="GlossWerk Term Scanner",
    page_icon="🔍",
    layout="wide",
)

st.title("GlossWerk Terminology Scanner")
st.markdown("Upload a German patent → review terminology → export glossary")


# --- Session state init ---
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "translations_done" not in st.session_state:
    st.session_state.translations_done = False
if "nouns" not in st.session_state:
    st.session_state.nouns = []
if "adjectives" not in st.session_state:
    st.session_state.adjectives = []
if "verbs" not in st.session_state:
    st.session_state.verbs = []
if "noun_translations" not in st.session_state:
    st.session_state.noun_translations = {}
if "adj_translations" not in st.session_state:
    st.session_state.adj_translations = {}
if "selected_nouns" not in st.session_state:
    st.session_state.selected_nouns = {}
if "selected_adjs" not in st.session_state:
    st.session_state.selected_adjs = {}


# --- Sidebar: API config ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
    )
    model = st.selectbox("Model", [
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-6",
    ])
    min_noun_freq = st.slider("Min noun frequency", 2, 20, 5)
    min_adj_freq = st.slider("Min adjective frequency", 1, 5, 2)

    st.divider()
    st.header("Pre-selected Terms")
    st.markdown("Paste existing glossary terms (one per line: `German\\tEnglish`)")
    preselected_text = st.text_area(
        "Existing glossary",
        placeholder="Stent\tstent\nKatheter\tcatheter",
        height=150,
    )


# --- Parse pre-selected terms ---
def parse_preselected(text):
    terms = {}
    for line in text.strip().split("\n"):
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                terms[parts[0].strip()] = parts[1].strip()
    return terms


preselected = parse_preselected(preselected_text) if preselected_text.strip() else {}


# --- File upload ---
uploaded = st.file_uploader("Upload German patent (.docx)", type=["docx"])

if uploaded and not st.session_state.extraction_done:
    with st.spinner("Extracting terminology..."):
        # Save to temp file for python-docx
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Extract text
        paragraphs = extract_text_from_docx(tmp_path)
        sentences = extract_sentences(paragraphs)

        # Extract nouns — use spaCy if available, otherwise fall back to heuristic
        if HAS_SPACY:
            try:
                raw_nouns, _ = extract_nouns_spacy(sentences)
            except OSError:
                raw_nouns = extract_nouns_heuristic(sentences)
        else:
            raw_nouns = extract_nouns_heuristic(sentences)
        filtered_nouns = {k: v for k, v in raw_nouns.items() if v >= min_noun_freq}
        clusters = cluster_variants(filtered_nouns)
        sorted_nouns = sorted(clusters.items(), key=lambda x: -x[1]["frequency"])

        # Extract adjectives
        adjectives, adj_variants = extract_technical_adjectives(sentences, min_freq=min_adj_freq)
        sorted_adjs = sorted(adjectives.items(), key=lambda x: -x[1])

        # Extract verbs
        verbs = extract_patent_verbs(sentences, min_freq=3)
        sorted_verbs = sorted(verbs.items(), key=lambda x: -x[1]["frequency"])

        # Store in session
        st.session_state.nouns = sorted_nouns
        st.session_state.adjectives = sorted_adjs
        st.session_state.verbs = sorted_verbs
        st.session_state.adj_variants = adj_variants
        st.session_state.total_sentences = len(sentences)
        st.session_state.filename = uploaded.name
        st.session_state.extraction_done = True

        # Clean up
        os.unlink(tmp_path)

    st.rerun()


# --- Show extraction results & get translations ---
if st.session_state.extraction_done and not st.session_state.translations_done:
    st.success(
        f"Extracted **{len(st.session_state.nouns)}** nouns, "
        f"**{len(st.session_state.adjectives)}** technical adjectives, "
        f"**{len(st.session_state.verbs)}** patent verbs "
        f"from {st.session_state.total_sentences} sentences"
    )

    if preselected:
        st.info(f"{len(preselected)} pre-selected terms will be locked in the glossary")

    if st.button("Get English translation proposals", type="primary"):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar")
        else:
            with st.spinner("Generating translation proposals via Claude..."):
                client = anthropic.Anthropic(api_key=api_key)

                # --- Helper: parse JSON from Claude response ---
                def parse_json_response(text):
                    """Extract JSON array from Claude response, handling code blocks."""
                    # Try raw first
                    text = text.strip()
                    # Strip markdown code blocks
                    if "```" in text:
                        parts = text.split("```")
                        for part in parts[1:]:
                            candidate = part.strip()
                            if candidate.startswith("json"):
                                candidate = candidate[4:]
                            candidate = candidate.strip()
                            if candidate.startswith("["):
                                return json.loads(candidate)
                    # Try finding array directly
                    start = text.find("[")
                    end = text.rfind("]")
                    if start != -1 and end != -1:
                        return json.loads(text[start:end + 1])
                    return json.loads(text)

                # --- Noun translations ---
                # Filter out pre-selected terms
                nouns_to_translate = [
                    (term, data) for term, data in st.session_state.nouns
                    if term not in preselected
                ]
                if nouns_to_translate:
                    # Batch in groups of 40 to avoid response truncation
                    BATCH_SIZE = 40
                    batches = [
                        nouns_to_translate[i:i + BATCH_SIZE]
                        for i in range(0, len(nouns_to_translate), BATCH_SIZE)
                    ]
                    progress = st.progress(0, text=f"Translating nouns (0/{len(batches)} batches)...")

                    for batch_idx, batch in enumerate(batches):
                        noun_prompt = "Propose English translations for these German patent terms.\n"
                        noun_prompt += 'Return a JSON array. Each item: {"de": "...", "en": ["option1", "option2"], "avoid": "...", "confidence": "high|medium", "note": "..."}\n'
                        noun_prompt += 'Rules:\n'
                        noun_prompt += '- "avoid": a common mistranslation to warn against. Omit if not applicable.\n'
                        noun_prompt += '- "confidence": "high" if you are certain this is the standard patent translation (e.g. Vorrichtung → device). "medium" if you are inferring from compound parts or the term is ambiguous.\n'
                        noun_prompt += '- "note": only if there is an important distinction between options.\n\nTerms:\n'
                        for term, data in batch:
                            noun_prompt += f"- {term} ({data['frequency']}x)\n"

                        resp = client.messages.create(
                            model=model,
                            max_tokens=4096,
                            system="You are a DE→EN patent terminology expert. For each German term, propose 2-3 English translations used in patent literature. Rank the most conventional translation first. For compound nouns, translate the full compound. Prefer technical terms over literal translations. Be honest about confidence — mark 'high' only for terms you know are standard in EPO/USPTO filings, 'medium' when inferring from component parts. Return ONLY a valid JSON array, no other text.",
                            messages=[{"role": "user", "content": noun_prompt}],
                        )
                        try:
                            noun_data = parse_json_response(resp.content[0].text)
                            for item in noun_data:
                                st.session_state.noun_translations[item["de"]] = {
                                    "options": item["en"],
                                    "avoid": item.get("avoid", ""),
                                    "confidence": item.get("confidence", "medium"),
                                    "note": item.get("note", ""),
                                }
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            st.warning(f"Batch {batch_idx + 1} parse error: {e}")

                        progress.progress(
                            (batch_idx + 1) / len(batches),
                            text=f"Translating nouns ({batch_idx + 1}/{len(batches)} batches)..."
                        )
                    progress.empty()

                # --- Adjective translations ---
                adjs_to_translate = [
                    (adj, freq) for adj, freq in st.session_state.adjectives
                    if adj not in preselected
                ]
                if adjs_to_translate:
                    adj_prompt = "Propose English translations for these German technical adjectives from a patent.\n"
                    adj_prompt += "Return a JSON array. Each item: {\"de\": \"...\", \"en\": \"correct translation\", \"avoid\": \"literal translation to avoid\", \"note\": \"...\"}\n\nAdjectives:\n"
                    for adj, freq in adjs_to_translate:
                        adj_prompt += f"- {adj} (appears {freq} times)\n"

                    resp = client.messages.create(
                        model=model,
                        max_tokens=2048,
                        system="You are a DE→EN patent terminology expert. For each German technical adjective, provide the correct patent translation and the literal translation to AVOID. Example: körpereigene → 'endogenous' (AVOID: 'body\\'s own'). Return ONLY valid JSON.",
                        messages=[{"role": "user", "content": adj_prompt}],
                    )
                    try:
                        adj_data = parse_json_response(resp.content[0].text)
                        for item in adj_data:
                            st.session_state.adj_translations[item["de"]] = {
                                "correct": item["en"],
                                "avoid": item.get("avoid", ""),
                                "note": item.get("note", ""),
                            }
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        st.error(f"Failed to parse adjective translations: {e}")

                st.session_state.translations_done = True
                st.rerun()


# --- Translation review interface ---
if st.session_state.translations_done:
    st.header(f"Terminology Review — {st.session_state.filename}")

    # --- Tab layout ---
    tab_nouns, tab_adjs, tab_verbs, tab_export = st.tabs([
        f"Nouns ({len(st.session_state.nouns)})",
        f"Adjectives ({len(st.session_state.adjectives)})",
        f"Verbs ({len(st.session_state.verbs)})",
        "Export Glossary",
    ])

    # --- NOUNS TAB ---
    with tab_nouns:
        st.subheader("Nouns — select preferred translation")
        st.markdown("*These will be enforced consistently throughout the translation.*")

        for term, data in st.session_state.nouns:
            freq = data["frequency"]
            variants = data.get("variants", [])
            variant_str = f" *(also: {', '.join(variants)})*" if variants else ""

            # Pre-selected term — locked
            if term in preselected:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{term}** ({freq}x){variant_str}")
                with col2:
                    st.text_input(
                        f"_{term}_locked",
                        value=preselected[term],
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"noun_locked_{term}",
                    )
                    st.caption("🔒 Pre-selected")
                st.session_state.selected_nouns[term] = preselected[term]
                continue

            # Translation available
            trans = st.session_state.noun_translations.get(term)
            if trans:
                # Confidence indicator
                conf = trans.get("confidence", "medium")
                conf_icon = "🟢" if conf == "high" else "🟡"

                col1, col2, col3 = st.columns([1, 1.2, 1])
                with col1:
                    st.markdown(f"**{term}** ({freq}x){variant_str}")
                    st.caption(f"{conf_icon} {conf} confidence")
                with col2:
                    options = list(trans["options"])  # copy to avoid mutation
                    options.append("✏️ Custom...")
                    choice = st.selectbox(
                        f"_{term}",
                        options,
                        label_visibility="collapsed",
                        key=f"noun_select_{term}",
                    )
                    if choice == "✏️ Custom...":
                        custom = st.text_input(
                            f"Custom translation for {term}",
                            label_visibility="collapsed",
                            key=f"noun_custom_{term}",
                        )
                        if custom:
                            st.session_state.selected_nouns[term] = custom
                    else:
                        st.session_state.selected_nouns[term] = choice
                with col3:
                    # Editable avoid field
                    avoid_default = trans.get("avoid", "")
                    avoid_val = st.text_input(
                        f"Avoid for {term}",
                        value=avoid_default,
                        placeholder="translation to avoid",
                        label_visibility="collapsed",
                        key=f"noun_avoid_{term}",
                    )
                    if avoid_val:
                        st.caption(f"~~{avoid_val}~~ AVOID")
                    if trans.get("note"):
                        st.caption(trans["note"])

    # --- ADJECTIVES TAB ---
    with tab_adjs:
        st.subheader("Technical Adjectives — select preferred translation")
        st.markdown("*These will be enforced consistently. The 'AVOID' column shows the literal translation that is wrong in patent context.*")

        for adj, freq in st.session_state.adjectives:
            # Pre-selected — locked
            if adj in preselected:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.markdown(f"**{adj}** ({freq}x)")
                with col2:
                    st.text_input(f"_{adj}", value=preselected[adj], disabled=True,
                                  label_visibility="collapsed", key=f"adj_locked_{adj}")
                    st.caption("🔒 Pre-selected")
                st.session_state.selected_adjs[adj] = preselected[adj]
                continue

            trans = st.session_state.adj_translations.get(adj)
            if trans:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.markdown(f"**{adj}** ({freq}x)")
                with col2:
                    default = trans["correct"]
                    custom = st.text_input(
                        f"_{adj}",
                        value=default,
                        label_visibility="collapsed",
                        key=f"adj_input_{adj}",
                    )
                    st.session_state.selected_adjs[adj] = custom
                with col3:
                    avoid_default = trans.get("avoid", "")
                    avoid_val = st.text_input(
                        f"Avoid for {adj}",
                        value=avoid_default,
                        placeholder="translation to avoid",
                        label_visibility="collapsed",
                        key=f"adj_avoid_{adj}",
                    )
                    if avoid_val:
                        st.caption(f"~~{avoid_val}~~ AVOID")
                    if trans.get("note"):
                        st.caption(trans["note"])

    # --- VERBS TAB ---
    with tab_verbs:
        st.subheader("Patent Verbs — reference only")
        st.markdown("*These are NOT enforced in the glossary. They need different translations depending on context. Use this as a reference sheet.*")

        for verb, info in st.session_state.verbs:
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col1:
                st.markdown(f"**{verb}** ({info['frequency']}x)")
            with col2:
                st.markdown(f"→ {info['translations']}")
            with col3:
                forms = ", ".join(info["forms_found"])
                st.caption(f"Forms: {forms}")

    # --- EXPORT TAB ---
    with tab_export:
        st.subheader("Export Glossary")

        # Build glossary from selections
        glossary_lines = []

        # Nouns
        for term, translation in st.session_state.selected_nouns.items():
            if translation and translation != "✏️ Custom...":
                glossary_lines.append(f"{term}\t{translation}")

        # Adjectives
        for adj, translation in st.session_state.selected_adjs.items():
            if translation:
                glossary_lines.append(f"{adj}\t{translation}")

        glossary_text = "\n".join(glossary_lines)

        st.metric("Total glossary entries", len(glossary_lines))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Glossary preview:**")
            st.code(glossary_text if glossary_text else "(no terms selected yet)")

        with col2:
            st.markdown("**Verb reference:**")
            verb_ref = ""
            for verb, info in st.session_state.verbs:
                verb_ref += f"{verb} → {info['translations']}\n"
                verb_ref += f"  Forms: {', '.join(info['forms_found'])}\n\n"
            st.code(verb_ref if verb_ref else "(no verbs detected)")

        st.divider()

        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download Glossary (TSV)",
                data=glossary_text,
                file_name=f"{st.session_state.filename.replace('.docx', '')}_glossary.tsv",
                mime="text/tab-separated-values",
                type="primary",
            )
        with col2:
            st.download_button(
                "Download Verb Reference",
                data=verb_ref,
                file_name=f"{st.session_state.filename.replace('.docx', '')}_verbs.md",
                mime="text/markdown",
            )
        with col3:
            # Full JSON export
            full_export = {
                "source_file": st.session_state.filename,
                "glossary": {term: trans for term, trans in st.session_state.selected_nouns.items()},
                "adjective_glossary": {adj: trans for adj, trans in st.session_state.selected_adjs.items()},
                "verb_reference": {verb: info for verb, info in st.session_state.verbs},
            }
            st.download_button(
                "Download Full JSON",
                data=json.dumps(full_export, ensure_ascii=False, indent=2),
                file_name=f"{st.session_state.filename.replace('.docx', '')}_terminology.json",
                mime="application/json",
            )


# --- Reset button ---
if st.session_state.extraction_done:
    st.divider()
    if st.button("Reset — scan a different patent"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
