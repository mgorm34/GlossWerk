"""
GlossWerk — DE→EN Patent Translation Pipeline

Upload a German patent → scan & review terminology → translate with QE →
review & edit side-by-side → export.

Run: streamlit run glosswerk_app.py
Requires: ANTHROPIC_API_KEY environment variable
Install: pip install anthropic streamlit python-docx spacy
         python -m spacy download de_core_news_lg
"""

import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import streamlit as st
from docx.shared import Pt

# --- Path setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Module imports ---
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "extract_terms",
    os.path.join(PROJECT_ROOT, "skills", "glosswerk-term-scanner", "scripts", "extract_terms.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_text_from_docx_terms = _mod.extract_text_from_docx
extract_sentences = _mod.extract_sentences
extract_nouns_spacy = _mod.extract_nouns_spacy
extract_nouns_heuristic = _mod.extract_nouns_heuristic
extract_technical_adjectives = _mod.extract_technical_adjectives
extract_patent_verbs = _mod.extract_patent_verbs
HAS_SPACY = _mod.HAS_SPACY

_spec2 = importlib.util.spec_from_file_location(
    "analyze_structure",
    os.path.join(PROJECT_ROOT, "skills", "glosswerk-structural-analyzer", "scripts", "analyze_structure.py"),
)
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
analyze_document = _mod2.analyze_document

from translate import (
    translate_document, extract_text_from_docx as extract_text_translate,
    split_sentences as split_sentences_translate,
)
from quality_estimate import (
    evaluate_translations, compute_triage,
    load_training_pairs, select_few_shot_examples,
)
from assemble import assemble_document
from demo_auth import show_auth_gate, record_patent_use, validate_code, WATERMARK_TEXT

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Demo mode toggle — set to False for local dev without auth
DEMO_MODE = os.environ.get("GLOSSWERK_DEMO", "false").lower() == "true"


@st.cache_resource
def load_nlp():
    try:
        import spacy
        return spacy.load("de_core_news_lg")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="GlossWerk", page_icon="G", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .triage-card { padding: 1rem 1.2rem; border-radius: 10px; text-align: center; font-weight: 600; }
    .triage-green { background: #d1fae5; color: #065f46; border: 2px solid #6ee7b7; }
    .triage-orange { background: #fef3c7; color: #92400e; border: 2px solid #fcd34d; }
    .triage-red { background: #fee2e2; color: #991b1b; border: 2px solid #fca5a5; }

    .conf-high { color: #059669; font-weight: 600; }
    .conf-medium { color: #d97706; font-weight: 600; }
    .conf-low { color: #dc2626; font-weight: 600; }

    .qe-note { font-size: 0.88rem; padding: 0.6rem 0.8rem; border-radius: 6px; margin-top: 0.4rem; color: #1f2937; }
    .qe-suggestion { font-size: 1rem; line-height: 1.6; margin-top: 0.5rem; padding: 0.5rem 0; }
    .qe-suggestion strong { background: #fef08a; padding: 1px 3px; border-radius: 2px; }
    .qe-minor { background: #fef9c3; border-left: 3px solid #eab308; color: #713f12; }
    .qe-major { background: #fee2e2; border-left: 3px solid #ef4444; color: #7f1d1d; }
    .qe-critical { background: #fecaca; border-left: 3px solid #b91c1c; color: #7f1d1d; }
    .qe-good { background: #d1fae5; border-left: 3px solid #10b981; color: #065f46; }

    .source-box {
        font-size: 0.9rem; color: #374151; padding: 0.6rem;
        background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;
        min-height: 60px; line-height: 1.5;
    }
    .locked-box {
        font-size: 0.9rem; color: #065f46; padding: 0.6rem;
        background: #f0fdf4; border-radius: 6px; border: 1px solid #6ee7b7;
        min-height: 60px; line-height: 1.5;
    }

    .reasoning-text { font-size: 0.8rem; color: #6b7280; font-style: italic; margin-top: 0.2rem; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>GlossWerk</h1>
    <p>DE → EN Patent Translation Pipeline</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════

demo_auth = None
if DEMO_MODE:
    demo_auth = show_auth_gate()
    if not demo_auth:
        st.stop()
    # Show demo status bar
    st.markdown(
        f"<div style='background:#f0fdf4; padding:0.5rem 1rem; border-radius:8px; "
        f"font-size:0.85rem; color:#065f46; border:1px solid #6ee7b7; margin-bottom:1rem;'>"
        f"Demo — <strong>{demo_auth['company']}</strong> · "
        f"{demo_auth['patents_remaining']} patents remaining · "
        f"{demo_auth['days_remaining']} days left</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Settings")
    # In demo mode, API key is server-side — hide from user
    if DEMO_MODE:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = "claude-sonnet-4-6"
    else:
        api_key = st.text_input("API Key", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-6", "claude-opus-4-6",
                                        "claude-haiku-4-5-20251001"], index=0)

    with st.expander("Advanced", expanded=False):
        min_adj_freq = st.slider("Min adjective frequency", 1, 10, 2)
        batch_size_trans = st.slider("Translation batch size", 10, 100, 50)
        batch_size_qe = st.slider("QE batch size", 5, 40, 15)
        n_few_shot = st.slider("Few-shot QE examples", 0, 50, 30)
        training_pairs_path = st.text_input(
            "Training pairs",
            value=os.path.join(PROJECT_ROOT, "data", "hter_training", "training_pairs.jsonl"),
        )

    st.divider()
    st.markdown("### Document")
    uploaded_file = st.file_uploader("German patent (.docx)", type=["docx"])
    glossary_upload = st.file_uploader("Glossary TSV (optional)", type=["tsv", "txt"])
    pre_selected = st.text_area("Existing glossary (DE\\tEN per line)", height=80)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

_defaults = {
    "raw_text": None, "sentences": None, "structural_analysis": None,
    "glossary": {}, "noun_counts": None, "adj_counts": None,
    "adj_variants": None, "verb_info": None, "lemma_map": None,
    "noun_proposals": {}, "adj_proposals": {},
    "translations": None, "qe_results": None, "triage": None,
    "confirmed": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, dict) else dict(v)

# Parse pre-selected / uploaded glossary
pre_glossary = {}
if pre_selected and pre_selected.strip():
    for line in pre_selected.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            pre_glossary[parts[0].strip()] = parts[1].strip()

if glossary_upload:
    content = glossary_upload.read().decode("utf-8")
    for line in content.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            pre_glossary[parts[0].strip()] = parts[1].strip()

for de, en in pre_glossary.items():
    if de not in st.session_state.glossary:
        st.session_state.glossary[de] = en

if uploaded_file is None:
    st.info("Upload a German .docx patent in the sidebar to get started.")
    st.stop()

# Ensure file buffer is at start before reading (Streamlit reruns re-use the object)
uploaded_file.seek(0)
with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
    tmp.write(uploaded_file.read())
    docx_path = tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: term translation with alternatives
# ══════════════════════════════════════════════════════════════════════════════

def translate_terms_batch(terms, term_type="noun"):
    """Get EN translations with alternatives for a list of DE terms."""
    if not terms or not api_key or not HAS_ANTHROPIC:
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    all_proposals = {}

    if term_type == "noun":
        schema = (
            '{"de": "German", "en": "best translation", '
            '"alternatives": ["alt1", "alt2"], '
            '"confidence": "high|medium|low", '
            '"reasoning": "brief explanation only if confidence is NOT high"}'
        )
        instruction = (
            "You are a DE→EN patent terminology expert. "
            "For each German noun, provide: the best English translation for patent context, "
            "1-2 alternatives if they exist, confidence level, and brief reasoning ONLY if "
            "confidence is medium or low. Return a JSON array."
        )
    else:  # adjective
        schema = (
            '{"de": "German", "en": "best translation", '
            '"alternatives": ["alt1", "alt2"], '
            '"avoid": "common mistranslation to avoid (only if confident)", '
            '"confidence": "high|medium|low", '
            '"reasoning": "brief explanation of why literal translation is wrong"}'
        )
        instruction = (
            "You are a DE→EN patent terminology expert. "
            "For each German technical adjective, provide: the best English translation, "
            "1-2 alternatives, an 'avoid' mistranslation ONLY if the literal translation is "
            "definitely wrong in patent context, confidence level, and reasoning. Return a JSON array."
        )

    for batch_start in range(0, len(terms), 40):
        batch = terms[batch_start:batch_start + 40]
        terms_json = json.dumps(batch, ensure_ascii=False)

        prompt = f"{instruction}\n\nSchema per item: {schema}\n\nTerms: {terms_json}"

        try:
            msg = client.messages.create(model=model, max_tokens=8192,
                                         messages=[{"role": "user", "content": prompt}])
            raw = msg.content[0].text.strip()
            code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
            if code_match:
                raw = code_match.group(1).strip()
            bs = raw.find('[')
            be = raw.rfind(']')
            if bs >= 0 and be > bs:
                parsed = json.loads(raw[bs:be + 1])
                for item in parsed:
                    all_proposals[item["de"]] = item
        except Exception as e:
            pass

    return all_proposals


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_terms, tab_translate, tab_review, tab_export = st.tabs([
    "Terminology", "Translate & QE", "Review", "Export"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: TERMINOLOGY
# ══════════════════════════════════════════════════════════════════════════════

with tab_terms:

    # Pre-compute sentence count for the frequency slider
    if "doc_sentence_count" not in st.session_state:
        _raw = extract_text_from_docx_terms(docx_path)
        _sents = extract_sentences(_raw)
        st.session_state.doc_sentence_count = len(_sents)

    n_sents = st.session_state.doc_sentence_count
    auto_default = max(2, min(8, round(n_sents / 20)))
    slider_max = max(auto_default + 3, 10)

    st.markdown(f"**Noun frequency threshold** — {n_sents} sentences detected")
    freq_col, help_col = st.columns([3, 2])
    with freq_col:
        min_noun_freq = st.slider(
            "Min appearances to include a noun", min_value=1,
            max_value=slider_max, value=auto_default,
            label_visibility="collapsed",
        )
    with help_col:
        st.caption(
            f"📏 Recommended: **{auto_default}** for this patent length. "
            f"Lower = more terms (noisier). Higher = only high-frequency core terms."
        )

    if st.button("Scan & Translate Terminology", type="primary", use_container_width=True):
        progress = st.progress(0, text="Extracting terms...")

        raw_text = extract_text_from_docx_terms(docx_path)
        sentences = extract_sentences(raw_text)
        st.session_state.raw_text = raw_text
        st.session_state.sentences = sentences

        # Diagnostic: detect if text looks like English instead of German
        if sentences:
            sample = " ".join(sentences[:5])
            mid_caps = sum(1 for w in sample.split() if w[0].isupper() and w not in ("FIG.", "Fig."))
            total_words = len(sample.split())
            cap_ratio = mid_caps / total_words if total_words else 0
            if cap_ratio < 0.15:
                st.warning(
                    f"⚠️ Low capitalization ratio ({cap_ratio:.0%}) — this text may be English, "
                    f"not German. The term extractor requires German source text. "
                    f"First 100 chars: `{sample[:100]}…`"
                )

        progress.progress(0.15, text="Extracting nouns...")

        if HAS_SPACY:
            try:
                noun_counts, lemma_map = extract_nouns_spacy(sentences)
            except Exception:
                noun_counts = extract_nouns_heuristic(sentences)
                lemma_map = {}
        else:
            noun_counts = extract_nouns_heuristic(sentences)
            lemma_map = {}
        noun_counts = {k: v for k, v in noun_counts.items() if v >= min_noun_freq}

        progress.progress(0.25, text="Extracting adjectives & verbs...")
        adj_freq, adj_variants = extract_technical_adjectives(sentences, min_freq=min_adj_freq)
        verb_info = extract_patent_verbs(sentences)

        st.session_state.noun_counts = noun_counts
        st.session_state.adj_counts = adj_freq
        st.session_state.adj_variants = adj_variants
        st.session_state.verb_info = verb_info
        st.session_state.lemma_map = lemma_map

        # API translations: nouns
        untranslated_nouns = [n for n in noun_counts if n not in pre_glossary]
        if untranslated_nouns and api_key:
            progress.progress(0.35, text=f"Translating {len(untranslated_nouns)} nouns...")
            noun_proposals = translate_terms_batch(untranslated_nouns, "noun")
            st.session_state.noun_proposals = noun_proposals
            for de, info in noun_proposals.items():
                if de not in st.session_state.glossary:
                    st.session_state.glossary[de] = info.get("en", "")

        # API translations: adjectives
        untranslated_adjs = list(adj_freq.keys())
        if untranslated_adjs and api_key:
            progress.progress(0.65, text=f"Translating {len(untranslated_adjs)} adjectives...")
            adj_proposals = translate_terms_batch(untranslated_adjs, "adjective")
            st.session_state.adj_proposals = adj_proposals

        progress.progress(1.0, text="Done!")
        st.success(f"{len(noun_counts)} nouns · {len(adj_freq)} adjectives · {len(verb_info)} verbs")

    # --- Results in sub-tabs ---
    if st.session_state.noun_counts is not None:
        term_tab1, term_tab2, term_tab3 = st.tabs(["Nouns", "Technical Adjectives", "Patent Verbs"])

        # ---- NOUNS ----
        with term_tab1:
            nouns = st.session_state.noun_counts
            proposals = st.session_state.noun_proposals or {}

            st.caption(f"{len(nouns)} terms — select preferred translation or type your own")

            # Header row
            hc1, hc2, hc3, hc4 = st.columns([2.5, 3, 2, 2])
            hc1.markdown("**German term**")
            hc2.markdown("**English translation**")
            hc3.markdown("**Confidence**")
            hc4.markdown("**Reasoning**")
            st.divider()

            for noun, count in sorted(nouns.items(), key=lambda x: -x[1]):
                locked = noun in pre_glossary
                info = proposals.get(noun, {})
                current_en = st.session_state.glossary.get(noun, info.get("en", ""))
                confidence = info.get("confidence", "")
                alternatives = info.get("alternatives", [])
                reasoning = info.get("reasoning", "")

                col_de, col_en, col_conf, col_reason = st.columns([2.5, 3, 2, 2])

                with col_de:
                    st.markdown(f"{'🔒 ' if locked else ''}**{noun}** ({count}x)")

                with col_en:
                    if locked:
                        st.text(current_en)
                    else:
                        # Build options: current + alternatives + custom
                        options = [current_en] if current_en else []
                        for alt in alternatives:
                            if alt and alt not in options:
                                options.append(alt)
                        if not options:
                            options = [""]
                        options.append("✏️ Custom...")

                        selected = st.selectbox(
                            f"sel_{noun}", options=options,
                            label_visibility="collapsed", key=f"noun_sel_{noun}",
                        )

                        if selected == "✏️ Custom...":
                            custom = st.text_input(
                                f"custom_{noun}", value="", key=f"noun_custom_{noun}",
                                label_visibility="collapsed", placeholder="Type translation..."
                            )
                            if custom:
                                st.session_state.glossary[noun] = custom
                        elif selected:
                            st.session_state.glossary[noun] = selected

                with col_conf:
                    if confidence:
                        css_class = f"conf-{confidence}"
                        st.markdown(f"<span class='{css_class}'>● {confidence}</span>",
                                    unsafe_allow_html=True)

                with col_reason:
                    if reasoning and confidence in ("medium", "low"):
                        st.markdown(f"<span class='reasoning-text'>{reasoning}</span>",
                                    unsafe_allow_html=True)

            # --- Manual glossary entry ---
            st.divider()
            st.markdown("**Add custom term**")
            add_col1, add_col2, add_col3 = st.columns([3, 3, 1])
            with add_col1:
                new_de = st.text_input("German term", key="add_de",
                                       label_visibility="collapsed",
                                       placeholder="German term...")
            with add_col2:
                new_en = st.text_input("English translation", key="add_en",
                                       label_visibility="collapsed",
                                       placeholder="English translation...")
            with add_col3:
                if st.button("➕", key="add_glossary_btn", use_container_width=True):
                    if new_de and new_en:
                        st.session_state.glossary[new_de.strip()] = new_en.strip()
                        # Also add to noun_counts so it shows in the list
                        if st.session_state.noun_counts is not None:
                            if new_de.strip() not in st.session_state.noun_counts:
                                st.session_state.noun_counts[new_de.strip()] = 0
                        st.rerun()

        # ---- ADJECTIVES ----
        with term_tab2:
            adjs = st.session_state.adj_counts or {}
            adj_proposals = st.session_state.adj_proposals or {}

            st.caption(f"{len(adjs)} technical adjectives — review translations and avoid notes")

            hc1, hc2, hc3, hc4 = st.columns([2.5, 3, 2, 2])
            hc1.markdown("**German adjective**")
            hc2.markdown("**English translation**")
            hc3.markdown("**Avoid**")
            hc4.markdown("**Reasoning**")
            st.divider()

            for adj, count in sorted(adjs.items(), key=lambda x: -x[1]):
                info = adj_proposals.get(adj, {})
                en_val = info.get("en", "")
                alternatives = info.get("alternatives", [])
                avoid_val = info.get("avoid", "")
                reasoning = info.get("reasoning", "")
                confidence = info.get("confidence", "")

                col_de, col_en, col_avoid, col_reason = st.columns([2.5, 3, 2, 2])

                with col_de:
                    conf_class = f"conf-{confidence}" if confidence else ""
                    conf_dot = f"<span class='{conf_class}'>● </span>" if confidence else ""
                    st.markdown(f"{conf_dot}**{adj}** ({count}x)", unsafe_allow_html=True)

                with col_en:
                    options = [en_val] if en_val else []
                    for alt in alternatives:
                        if alt and alt not in options:
                            options.append(alt)
                    if not options:
                        options = [""]
                    options.append("✏️ Custom...")

                    selected = st.selectbox(
                        f"adj_sel_{adj}", options=options,
                        label_visibility="collapsed", key=f"adj_sel_{adj}",
                    )
                    if selected == "✏️ Custom...":
                        custom = st.text_input(
                            f"adj_custom_{adj}", value="", key=f"adj_custom_{adj}",
                            label_visibility="collapsed", placeholder="Type translation..."
                        )
                        if custom:
                            st.session_state.glossary[adj] = custom
                    elif selected:
                        st.session_state.glossary[adj] = selected

                with col_avoid:
                    if avoid_val:
                        st.markdown(f"~~{avoid_val}~~")

                with col_reason:
                    if reasoning:
                        st.markdown(f"<span class='reasoning-text'>{reasoning}</span>",
                                    unsafe_allow_html=True)

        # ---- VERBS ----
        with term_tab3:
            verbs = st.session_state.verb_info or {}

            st.caption(f"{len(verbs)} patent verbs — add/remove translations individually")

            # Initialize verb_translations in session state from existing data
            if "verb_translations" not in st.session_state:
                st.session_state.verb_translations = {}
                for v, vinfo in verbs.items():
                    raw = vinfo.get("translations", "")
                    # Parse existing slash-separated into list
                    items = [t.strip() for t in raw.replace("/", ",").split(",") if t.strip()]
                    st.session_state.verb_translations[v] = items

            hc1, hc2, hc3 = st.columns([2.5, 4.5, 2])
            hc1.markdown("**German verb**")
            hc2.markdown("**Translations**")
            hc3.markdown("**Forms found**")
            st.divider()

            for verb, info in sorted(verbs.items(), key=lambda x: -x[1]["frequency"]):
                col_v, col_t, col_f = st.columns([2.5, 4.5, 2])

                # Ensure this verb has a list
                if verb not in st.session_state.verb_translations:
                    raw = info.get("translations", "")
                    st.session_state.verb_translations[verb] = [
                        t.strip() for t in raw.replace("/", ",").split(",") if t.strip()
                    ]
                current_tags = st.session_state.verb_translations[verb]

                with col_v:
                    st.markdown(f"**{verb}** ({info['frequency']}x)")

                with col_t:
                    # Show existing translations as removable tags
                    if current_tags:
                        tag_cols = st.columns(min(len(current_tags) + 1, 6))
                        for ti, tag in enumerate(current_tags):
                            with tag_cols[ti % (len(tag_cols) - 1)] if len(tag_cols) > 1 else tag_cols[0]:
                                if st.button(f"✕ {tag}", key=f"rm_verb_{verb}_{ti}",
                                             type="secondary"):
                                    st.session_state.verb_translations[verb].pop(ti)
                                    # Sync back
                                    st.session_state.verb_info[verb]["translations"] = \
                                        ", ".join(st.session_state.verb_translations[verb])
                                    st.rerun()

                    # Add new translation
                    add_c1, add_c2 = st.columns([4, 1])
                    with add_c1:
                        new_trans = st.text_input(
                            f"add_verb_{verb}", value="",
                            label_visibility="collapsed", key=f"verb_add_{verb}",
                            placeholder="Add translation..."
                        )
                    with add_c2:
                        if st.button("➕", key=f"verb_add_btn_{verb}"):
                            if new_trans and new_trans.strip():
                                val = new_trans.strip()
                                if val not in st.session_state.verb_translations[verb]:
                                    st.session_state.verb_translations[verb].append(val)
                                    st.session_state.verb_info[verb]["translations"] = \
                                        ", ".join(st.session_state.verb_translations[verb])
                                    st.rerun()

                with col_f:
                    forms = info.get("forms_found", [])
                    if forms:
                        st.caption(", ".join(forms[:4]))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRANSLATE & QE
# ══════════════════════════════════════════════════════════════════════════════

with tab_translate:

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar.")
        st.stop()
    if not HAS_ANTHROPIC:
        st.error("Install anthropic: pip install anthropic")
        st.stop()

    if st.button("Translate & Evaluate", type="primary", use_container_width=True):
        # Demo mode: check remaining patents
        if DEMO_MODE:
            recheck = validate_code(st.session_state.demo_code)
            if not recheck["valid"]:
                st.error(recheck["message"])
                st.stop()
            record_patent_use(st.session_state.demo_code, uploaded_file.name)

        progress = st.progress(0, text="Starting pipeline...")
        status = st.empty()

        # Step 1: Structural analysis (silent)
        status.text("Analyzing sentence structure...")
        progress.progress(0.05, text="Structural analysis...")
        nlp = load_nlp()
        if nlp:
            structural = analyze_document(docx_path, nlp)
            st.session_state.structural_analysis = structural
        else:
            st.session_state.structural_analysis = None

        # Step 2: Translate
        raw_text = extract_text_translate(docx_path)
        sentences = split_sentences_translate(raw_text)
        glossary = st.session_state.glossary if st.session_state.glossary else None

        def trans_progress(current, total):
            pct = 0.1 + (current / total) * 0.5
            progress.progress(min(pct, 0.6), text=f"Translating {current}/{total}...")

        status.text("Translating...")
        results = translate_document(
            sentences=sentences, api_key=api_key, model=model,
            glossary=glossary, structural_analysis=st.session_state.structural_analysis,
            batch_size=batch_size_trans, progress_callback=trans_progress,
        )

        st.session_state.translations = {
            "metadata": {
                "source_file": uploaded_file.name, "model": model,
                "n_sentences": len(sentences),
                "n_structural_hints": sum(1 for r in results if r["had_structural_hint"]),
                "glossary_terms": len(st.session_state.glossary),
            },
            "translations": results,
        }

        # Step 3: QE
        progress.progress(0.65, text="Quality estimation...")
        status.text("Evaluating quality...")

        few_shot = None
        if training_pairs_path and os.path.exists(training_pairs_path) and n_few_shot > 0:
            all_pairs = load_training_pairs(training_pairs_path)
            few_shot = select_few_shot_examples(all_pairs, n_few_shot)

        def qe_progress(current, total):
            pct = 0.65 + (current / total) * 0.3
            progress.progress(min(pct, 0.95), text=f"QE {current}/{total}...")

        qe_results = evaluate_translations(
            translations=results, api_key=api_key, model=model,
            few_shot_examples=few_shot, batch_size=batch_size_qe,
            progress_callback=qe_progress,
        )

        triage = compute_triage(qe_results)
        st.session_state.qe_results = qe_results
        st.session_state.triage = triage

        progress.progress(1.0, text="Complete!")
        status.empty()

    # --- Results ---
    if st.session_state.triage:
        triage = st.session_state.triage
        s = triage["summary"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="triage-card triage-green">
                <div style="font-size:2.2rem">{s['green_count']}</div>
                <div>Publishable ({s['green_pct']:.0f}%)</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="triage-card triage-orange">
                <div style="font-size:2.2rem">{s['orange_count']}</div>
                <div>Review ({s['orange_pct']:.0f}%)</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="triage-card triage-red">
                <div style="font-size:2.2rem">{s['red_count']}</div>
                <div>Edit needed ({s['red_pct']:.0f}%)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        error_breakdown = s.get("error_breakdown", {})
        if error_breakdown:
            cols = st.columns(min(len(error_breakdown), 6))
            for i, (etype, count) in enumerate(sorted(error_breakdown.items(), key=lambda x: -x[1])):
                with cols[i % len(cols)]:
                    st.metric(etype.title(), count)

        if s.get("high_risk_total", 0) > 0:
            hr_pct = round(s["high_risk_green"] / s["high_risk_total"] * 100) if s["high_risk_total"] else 0
            st.caption(f"Structural analysis: {s['high_risk_total']} high-risk sentences → "
                       f"{s['high_risk_green']} publishable ({hr_pct}%)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: REVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_review:

    if st.session_state.translations is None or st.session_state.qe_results is None:
        st.info("Run Translate & QE first.")
        st.stop()

    translations = st.session_state.translations["translations"]
    qe_by_idx = {r["index"]: r for r in st.session_state.qe_results}
    green_set = set(st.session_state.triage["green"]) if st.session_state.triage else set()
    confirmed = st.session_state.confirmed

    # Progress tracker
    total_segs = len(translations)
    n_confirmed = len(confirmed)
    n_green = len(green_set)
    # "Done" = confirmed + unconfirmed greens (publishable without edits)
    n_done = n_confirmed + sum(1 for i in green_set if i not in confirmed)
    pct_confirmed = int(n_confirmed / total_segs * 100) if total_segs else 0
    pct_done = int(n_done / total_segs * 100) if total_segs else 0
    needs_review = total_segs - n_done

    st.progress(pct_confirmed / 100, text=f"**{pct_confirmed}%** confirmed ({n_confirmed}/{total_segs})  ·  {needs_review} segments need review")

    # Filter controls
    col_filter, _ = st.columns([4, 1])
    with col_filter:
        view_mode = st.radio(
            "Show", ["Needs review", "All segments", "Red only", "Confirmed"],
            horizontal=True, label_visibility="collapsed",
        )

    st.divider()

    for trans in translations:
        idx = trans["index"]
        qe = qe_by_idx.get(idx, {})
        rating = qe.get("rating", "unknown")
        is_green = idx in green_set
        is_confirmed = idx in confirmed
        suggestion = qe.get("suggestion", "")
        explanation = qe.get("explanation", "")
        error_cat = qe.get("error_category", "")

        # Filter logic
        if view_mode == "Needs review" and (is_green or is_confirmed):
            continue
        elif view_mode == "Red only" and rating not in ("major", "critical"):
            continue
        elif view_mode == "Confirmed" and not is_confirmed:
            continue

        badge = {"good": "🟢", "minor": "🟡", "major": "🔴", "critical": "⛔"}.get(rating, "⚪")

        # --- Segment header ---
        header_col, action_col = st.columns([5, 2])
        with header_col:
            st.markdown(f"{badge} **Segment {idx}** — _{rating}_")
        with action_col:
            btn_cols = st.columns(2)
            if is_confirmed:
                with btn_cols[0]:
                    st.markdown("✅ Locked")
                with btn_cols[1]:
                    if st.button("Unlock", key=f"unlock_{idx}", type="secondary"):
                        del st.session_state.confirmed[idx]
                        st.rerun()
            else:
                with btn_cols[0]:
                    if st.button("Confirm", key=f"confirm_{idx}", type="secondary"):
                        edited = st.session_state.get(f"edit_{idx}", trans.get("translation", ""))
                        st.session_state.confirmed[idx] = edited
                        st.rerun()
                with btn_cols[1]:
                    # Apply QE suggestion — puts text in editor, does NOT lock
                    if suggestion and rating != "good":
                        already_applied = st.session_state.get(f"applied_{idx}", False)
                        if already_applied:
                            st.markdown("✅ Fix applied")
                        else:
                            if st.button("Apply fix", key=f"apply_{idx}", type="primary"):
                                clean_suggestion = re.sub(r'\*\*(.+?)\*\*', r'\1', suggestion)
                                st.session_state[f"edit_{idx}"] = clean_suggestion
                                st.session_state[f"applied_{idx}"] = True
                                st.rerun()

        # --- Side by side ---
        col_de, col_en = st.columns(2)

        with col_de:
            st.caption("German source")
            st.markdown(f"<div class='source-box'>{trans.get('source', '')}</div>",
                        unsafe_allow_html=True)

        with col_en:
            st.caption("English translation")
            if is_confirmed:
                st.markdown(f"<div class='locked-box'>{confirmed[idx]}</div>",
                            unsafe_allow_html=True)
            else:
                current_text = trans.get("translation", "")
                st.text_area(
                    f"edit_{idx}", value=current_text,
                    label_visibility="collapsed", key=f"edit_{idx}",
                    height=100,
                )

        # --- QE reasoning ---
        if rating != "good" and explanation:
            qe_class = {"major": "qe-major", "critical": "qe-critical"}.get(rating, "qe-minor")
            suggestion_html = ""
            if suggestion:
                # Convert **bold** markers to <strong> for HTML rendering
                rendered = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', suggestion)
                suggestion_html = (
                    f"<div class='qe-suggestion'>"
                    f"<strong>Suggested:</strong> {rendered}"
                    f"</div>"
                )
            st.markdown(
                f"<div class='qe-note {qe_class}'>"
                f"<strong>{error_cat}:</strong> {explanation}"
                f"{suggestion_html}"
                f"</div>", unsafe_allow_html=True
            )
        elif rating == "good":
            st.markdown(
                f"<div class='qe-note qe-good'>No issues detected</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:

    if st.session_state.translations is None or st.session_state.qe_results is None:
        st.info("Run Translate & QE first.")
        st.stop()

    st.markdown("### QE Review Document")
    st.caption("Color-coded .docx with triage summary, source text, and QE annotations")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_source = st.checkbox("Include German source", value=True)
    with col_opt2:
        include_annotations = st.checkbox("Include QE notes", value=True)

    if st.button("Generate QE Document", type="primary"):
        with st.spinner("Building document..."):
            qe_data = {
                "qe_results": st.session_state.qe_results,
                "triage": st.session_state.triage,
            }
            output_path = os.path.join(tempfile.gettempdir(), "glosswerk_qe_review.docx")
            assemble_document(
                translations_data=st.session_state.translations,
                qe_data=qe_data, output_path=output_path,
                include_source=include_source,
                include_annotations=include_annotations,
            )
            with open(output_path, "rb") as f:
                st.download_button(
                    "Download QE Review Document", data=f.read(),
                    file_name=f"glosswerk_qe_{uploaded_file.name}",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

    st.divider()

    st.markdown("### Translation Only")
    st.caption("Clean English document — uses confirmed edits where available")

    if st.button("Generate Translation Document"):
        with st.spinner("Building document..."):
            from docx import Document as DocxDoc

            doc = DocxDoc()
            style = doc.styles['Normal']
            style.font.name = 'Calibri'
            style.font.size = Pt(10)

            for trans in st.session_state.translations["translations"]:
                idx = trans["index"]
                text = st.session_state.confirmed.get(idx, trans.get("translation", ""))
                if text:
                    doc.add_paragraph(text)

            # Demo watermark
            if DEMO_MODE:
                from docx.shared import RGBColor
                wp = doc.add_paragraph()
                wp.alignment = 1  # center
                run = wp.add_run(WATERMARK_TEXT)
                run.font.size = Pt(8)
                run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
                run.font.italic = True

            out_path = os.path.join(tempfile.gettempdir(), "glosswerk_translation.docx")
            doc.save(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download Translation", data=f.read(),
                    file_name=f"translation_{uploaded_file.name}",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

    st.divider()

    st.markdown("### Data Exports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Translations (JSON)",
            data=json.dumps(st.session_state.translations, ensure_ascii=False, indent=2),
            file_name="translations.json", mime="application/json",
        )
    with col2:
        qe_export = {"qe_results": st.session_state.qe_results, "triage": st.session_state.triage}
        st.download_button(
            "QE Results (JSON)",
            data=json.dumps(qe_export, ensure_ascii=False, indent=2),
            file_name="qe_results.json", mime="application/json",
        )

    if st.session_state.glossary:
        st.divider()
        glossary_tsv = "\n".join(f"{de}\t{en}" for de, en in sorted(st.session_state.glossary.items()))
        st.download_button(
            "Glossary (TSV)", data=glossary_tsv,
            file_name="glossary.tsv", mime="text/tab-separated-values",
        )
