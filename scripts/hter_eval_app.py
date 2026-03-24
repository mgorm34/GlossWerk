"""
GlossWerk - Translator Evaluation Interface v3
Multi-term capture + term browser/editor.

Usage:
    streamlit run hter_eval_app.py
"""

import json
import os
import streamlit as st
import csv
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EVAL_FILE = os.path.join(DATA_DIR, "qe_pipeline_results.tsv")
SAVE_FILE = os.path.join(DATA_DIR, "hter_evaluations.json")
TERMS_FILE = os.path.join(DATA_DIR, "hter_captured_terms.tsv")

st.set_page_config(page_title="GlossWerk Evaluation", layout="wide")


@st.cache_data
def load_data(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_evaluations():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_evaluations(evals):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(evals, f, ensure_ascii=False, indent=2)


def get_all_terms(evals):
    """Extract all captured terms from evaluations."""
    terms = []
    for idx, ev in evals.items():
        for term in ev.get("terms", []):
            if term.get("de") or term.get("correct"):
                terms.append({
                    "German": term.get("de", ""),
                    "Wrong MT": term.get("wrong", ""),
                    "Correct EN": term.get("correct", ""),
                    "Sentence #": int(idx) + 1,
                })
    return terms


def save_all_terms(evals):
    terms = []
    for idx, ev in evals.items():
        for term in ev.get("terms", []):
            if term.get("de") and term.get("correct"):
                terms.append({
                    "de_term": term["de"],
                    "correct_en": term["correct"],
                    "wrong_mt": term.get("wrong", ""),
                    "sentence_id": idx,
                })
    if terms:
        with open(TERMS_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["de_term", "correct_en", "wrong_mt", "sentence_id"], delimiter="\t")
            writer.writeheader()
            for t in terms:
                writer.writerow(t)
    return len(terms)


def show_evaluate_page(data, evals):
    """Main evaluation interface."""

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "num_terms" not in st.session_state:
        st.session_state.num_terms = 1

    idx = st.session_state.idx

    # Current sentence
    row = data[idx]
    str_idx = str(idx)
    current_eval = evals.get(str_idx, {})

    # Header
    changed = row.get("Changed", "No")
    deepl_qe = row.get("DeepL_QE_Score", "N/A")
    gw_qe = row.get("GlossWerk_QE_Score", "N/A")
    gw_triage = row.get("GlossWerk_Triage", "N/A")
    status = "🟢" if gw_triage == "publish" else "🟡" if gw_triage == "review" else "🔴"

    st.markdown(f"### Sentence {idx + 1} / {len(data)}  {status} `{gw_triage}`  |  Changed: **{changed}**  |  DeepL QE: **{deepl_qe}**  |  GW QE: **{gw_qe}**")
    st.divider()

    # Stacked translations
    st.markdown("**🇩🇪 German Source**")
    st.info(row.get("German_Source", ""))

    st.markdown("**🔵 DeepL Output**")
    st.warning(row.get("DeepL_Output", ""))

    st.markdown("**🟢 GlossWerk Output**")
    st.success(row.get("GlossWerk_Output", ""))

    st.markdown("**📖 Human Reference**")
    st.text_area("ref", row.get("Human_Reference", ""), height=80, disabled=True, label_visibility="collapsed")

    st.divider()

    # Scoring
    st.markdown("### Evaluation")
    col1, col2, col3 = st.columns(3)

    with col1:
        pref_options = ["DeepL", "GlossWerk", "Equal"]
        default_pref = pref_options.index(current_eval.get("preferred", "Equal")) if current_eval.get("preferred") else 2
        preferred = st.radio("Which is better?", pref_options, index=default_pref, key=f"pref_{idx}", horizontal=True)

    with col2:
        edits_deepl = st.slider("Edits — DeepL (0=perfect, 5=rewrite)", 0, 5, value=current_eval.get("edits_deepl", 2), key=f"ed_{idx}")

    with col3:
        edits_gw = st.slider("Edits — GlossWerk (0=perfect, 5=rewrite)", 0, 5, value=current_eval.get("edits_gw", 2), key=f"eg_{idx}")

    st.divider()

    # Multi-term capture
    st.markdown("### Terminology Capture")
    st.caption("Capture as many terms as you spot. Click '+ Add another term' for more rows.")

    existing_terms = current_eval.get("terms", [])
    num_terms = st.session_state.num_terms

    if num_terms < 1:
        num_terms = 1

    term_entries = []
    for t in range(num_terms):
        existing = existing_terms[t] if t < len(existing_terms) else {}
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            de = st.text_input(f"German term #{t+1}", value=existing.get("de", ""), key=f"de_{idx}_{t}", placeholder="e.g., Rastaufnahme")
        with tcol2:
            wrong = st.text_input(f"Wrong MT #{t+1}", value=existing.get("wrong", ""), key=f"wr_{idx}_{t}", placeholder="e.g., rest picture")
        with tcol3:
            correct = st.text_input(f"Correct EN #{t+1}", value=existing.get("correct", ""), key=f"co_{idx}_{t}", placeholder="e.g., latching receptacle")
        if de or wrong or correct:
            term_entries.append({"de": de, "wrong": wrong, "correct": correct})

    if st.button("+ Add another term", key=f"add_{idx}"):
        st.session_state.num_terms += 1
        st.rerun()

    notes = st.text_area("Notes", value=current_eval.get("notes", ""), key=f"notes_{idx}", height=60)

    st.divider()

    # Navigation and save
    col_prev, col_save, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("← Previous", use_container_width=True) and idx > 0:
            st.session_state.idx = idx - 1
            st.session_state.num_terms = max(1, len(evals.get(str(idx - 1), {}).get("terms", [{"de": ""}])))
            st.rerun()

    with col_save:
        if st.button("💾 Save & Next", type="primary", use_container_width=True):
            evals[str_idx] = {
                "preferred": preferred,
                "edits_deepl": edits_deepl,
                "edits_gw": edits_gw,
                "terms": [t for t in term_entries if t.get("de") or t.get("correct")],
                "notes": notes,
            }
            save_evaluations(evals)

            if idx < len(data) - 1:
                st.session_state.idx = idx + 1
                next_eval = evals.get(str(idx + 1), {})
                st.session_state.num_terms = max(1, len(next_eval.get("terms", [{"de": ""}])))
                st.rerun()
            else:
                st.success("All sentences evaluated!")

    with col_next:
        if st.button("Skip →", use_container_width=True) and idx < len(data) - 1:
            st.session_state.idx = idx + 1
            st.session_state.num_terms = max(1, len(evals.get(str(idx + 1), {}).get("terms", [{"de": ""}])))
            st.rerun()

    # Running summary
    completed = len([e for e in evals.values() if e.get("preferred")])
    if completed > 0:
        st.divider()
        st.markdown("### Running Summary")
        pref_counts = {"DeepL": 0, "GlossWerk": 0, "Equal": 0}
        total_ed = 0
        total_eg = 0
        n_scored = 0

        for ev in evals.values():
            if ev.get("preferred"):
                pref_counts[ev["preferred"]] = pref_counts.get(ev["preferred"], 0) + 1
                total_ed += ev.get("edits_deepl", 0)
                total_eg += ev.get("edits_gw", 0)
                n_scored += 1

        if n_scored > 0:
            scol1, scol2, scol3, scol4, scol5 = st.columns(5)
            with scol1:
                st.metric("GW Preferred", f"{100*pref_counts['GlossWerk']/n_scored:.0f}%")
            with scol2:
                st.metric("DeepL Preferred", f"{100*pref_counts['DeepL']/n_scored:.0f}%")
            with scol3:
                st.metric("Equal", f"{100*pref_counts['Equal']/n_scored:.0f}%")
            with scol4:
                st.metric("Avg Edits DeepL", f"{total_ed/n_scored:.1f}")
            with scol5:
                st.metric("Avg Edits GW", f"{total_eg/n_scored:.1f}")


def show_terms_page(evals):
    """Term browser and editor."""
    st.header("Captured Terminology")

    terms = get_all_terms(evals)

    if not terms:
        st.info("No terms captured yet. Evaluate some sentences and capture terms as you go.")
        return

    df = pd.DataFrame(terms)

    # Stats
    st.markdown(f"**{len(df)} terms captured** from {df['Sentence #'].nunique()} sentences")

    st.divider()

    # Search/filter
    search = st.text_input("Search terms", placeholder="Type to filter...")
    if search:
        mask = (
            df["German"].str.contains(search, case=False, na=False) |
            df["Wrong MT"].str.contains(search, case=False, na=False) |
            df["Correct EN"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # Display as editable table
    st.markdown("### All Terms")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "German": st.column_config.TextColumn("German Term", width="large"),
            "Wrong MT": st.column_config.TextColumn("Wrong MT Output", width="large"),
            "Correct EN": st.column_config.TextColumn("Correct English", width="large"),
            "Sentence #": st.column_config.NumberColumn("Sentence #", width="small"),
        },
    )

    st.divider()

    # Duplicate check
    st.markdown("### Duplicate Check")
    if len(df) > 0:
        dupes = df[df.duplicated(subset=["German"], keep=False)].sort_values("German")
        if len(dupes) > 0:
            st.warning(f"{len(dupes)} entries with duplicate German terms — review for consistency:")
            st.dataframe(dupes, use_container_width=True, hide_index=True)
        else:
            st.success("No duplicate German terms found.")

    st.divider()

    # Export
    st.markdown("### Export")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to TSV (for terminology DB import)", use_container_width=True):
            save_evaluations(evals)
            n = save_all_terms(evals)
            st.success(f"Exported {n} terms to {TERMS_FILE}")
            st.code(f"python 07_build_terminology.py --source custom --file \"{TERMS_FILE}\"", language="bash")

    with col2:
        csv_data = df.to_csv(index=False, sep="\t")
        st.download_button(
            "Download as TSV",
            csv_data,
            file_name="glosswerk_terms.tsv",
            mime="text/tab-separated-values",
            use_container_width=True,
        )

    st.divider()

    # Term frequency from evaluation context
    st.markdown("### Most Common Wrong MT Terms")
    if len(df) > 0 and df["Wrong MT"].str.len().sum() > 0:
        wrong_counts = df["Wrong MT"].value_counts().head(20)
        if len(wrong_counts) > 0:
            st.bar_chart(wrong_counts)


def show_summary_page(data, evals):
    """Full evaluation summary."""
    st.header("Evaluation Summary")

    completed = len([e for e in evals.values() if e.get("preferred")])
    total = len(data)

    if completed == 0:
        st.info("No evaluations yet. Start evaluating on the Evaluate tab.")
        return

    st.markdown(f"**{completed} / {total}** sentences evaluated ({100*completed/total:.0f}%)")
    st.divider()

    # Aggregate stats
    pref_counts = {"DeepL": 0, "GlossWerk": 0, "Equal": 0}
    total_ed = 0
    total_eg = 0
    edits_deepl_list = []
    edits_gw_list = []
    n_scored = 0

    for ev in evals.values():
        if ev.get("preferred"):
            pref_counts[ev["preferred"]] = pref_counts.get(ev["preferred"], 0) + 1
            ed = ev.get("edits_deepl", 0)
            eg = ev.get("edits_gw", 0)
            total_ed += ed
            total_eg += eg
            edits_deepl_list.append(ed)
            edits_gw_list.append(eg)
            n_scored += 1

    # Preference breakdown
    st.markdown("### Preference")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("GlossWerk Preferred", f"{pref_counts['GlossWerk']} ({100*pref_counts['GlossWerk']/n_scored:.0f}%)")
    with col2:
        st.metric("DeepL Preferred", f"{pref_counts['DeepL']} ({100*pref_counts['DeepL']/n_scored:.0f}%)")
    with col3:
        st.metric("Equal", f"{pref_counts['Equal']} ({100*pref_counts['Equal']/n_scored:.0f}%)")

    st.divider()

    # Edit scores
    st.markdown("### Edit Effort")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Edits — DeepL", f"{total_ed/n_scored:.2f}")
    with col2:
        st.metric("Avg Edits — GlossWerk", f"{total_eg/n_scored:.2f}")
    with col3:
        diff = total_ed/n_scored - total_eg/n_scored
        st.metric("Edit Reduction", f"{diff:+.2f}", delta=f"{100*diff/max(0.01,total_ed/n_scored):.0f}% less editing")

    # Edit distribution
    if edits_deepl_list and edits_gw_list:
        st.markdown("### Edit Score Distribution")
        edit_df = pd.DataFrame({
            "DeepL": edits_deepl_list,
            "GlossWerk": edits_gw_list,
        })
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DeepL edit scores**")
            st.bar_chart(pd.Series(edits_deepl_list).value_counts().sort_index())
        with col2:
            st.markdown("**GlossWerk edit scores**")
            st.bar_chart(pd.Series(edits_gw_list).value_counts().sort_index())

    st.divider()

    # Terms summary
    terms = get_all_terms(evals)
    st.markdown(f"### Terminology: {len(terms)} terms captured")

    st.divider()

    # Product demo numbers
    st.markdown("### Product Demo Numbers")
    st.markdown("*Use these in your pitch and LinkedIn updates:*")
    st.code(f"""GlossWerk Human Evaluation Results ({n_scored} patent sentences, DE→EN)

Preference:
  GlossWerk preferred: {100*pref_counts['GlossWerk']/n_scored:.0f}%
  DeepL preferred:     {100*pref_counts['DeepL']/n_scored:.0f}%
  Equal:               {100*pref_counts['Equal']/n_scored:.0f}%

Post-editing effort (0=perfect, 5=rewrite):
  Avg edits DeepL:     {total_ed/n_scored:.2f}
  Avg edits GlossWerk: {total_eg/n_scored:.2f}
  Reduction:           {100*diff/max(0.01,total_ed/n_scored):.0f}%

Terminology entries captured: {len(terms)}
""")


def main():
    st.title("GlossWerk — Translator Evaluation")

    if not os.path.exists(EVAL_FILE):
        st.error(f"No evaluation data found at {EVAL_FILE}")
        st.info("Run 11_quality_estimation.py first to generate the data.")
        return

    data = load_data(EVAL_FILE)
    evals = load_evaluations()

    # Sidebar navigation
    with st.sidebar:
        st.header("Progress")
        completed = len([e for e in evals.values() if e.get("preferred")])
        st.progress(completed / len(data) if data else 0)
        st.write(f"**{completed} / {len(data)}** evaluated")

        terms_captured = sum(len(e.get("terms", [])) for e in evals.values() if e.get("terms"))
        st.write(f"**{terms_captured}** terms captured")

        st.divider()

        page = st.radio("View", ["Evaluate", "Terms", "Summary"], label_visibility="collapsed")

        if page == "Evaluate":
            st.divider()
            st.header("Navigate")

            jump = st.number_input("Go to #", min_value=1, max_value=len(data), value=st.session_state.get("idx", 0) + 1, step=1)
            if st.button("Go", use_container_width=True):
                st.session_state.idx = jump - 1
                st.session_state.num_terms = max(1, len(evals.get(str(jump - 1), {}).get("terms", [{"de": ""}])))
                st.rerun()

            if st.button("Next unevaluated", use_container_width=True):
                for i in range(len(data)):
                    if str(i) not in evals or not evals[str(i)].get("preferred"):
                        st.session_state.idx = i
                        st.session_state.num_terms = 1
                        st.rerun()
                        break

    # Show selected page
    if page == "Evaluate":
        show_evaluate_page(data, evals)
    elif page == "Terms":
        show_terms_page(evals)
    elif page == "Summary":
        show_summary_page(data, evals)


if __name__ == "__main__":
    main()