# GlossWerk v3 — LLM-Based Patent Translation Pipeline

**Date:** March 2026
**Status:** Prototype phase — build to demo, evaluate interest, then decide

---

## What This Is

A DE→EN patent translation tool that does three things a raw LLM prompt cannot:

1. **Source-side structural analysis** — Parses German sentences before translation to detect high-risk constructions (deep clause nesting, verb-final chains, long pre-fields) and quantifies reordering difficulty. This computed signal guides both the translator and the QE layer.

2. **Document-level consistency enforcement** — Maintains terminology across the full patent by extracting German nouns, tracking their English translations, and enforcing majority-vote consistency with glossary override. Operates at document scope, not sentence scope.

3. **Calibrated triage** — Combines structural risk scores with LLM-based quality assessment to sort segments into publish/review/full-edit buckets, giving translators a prioritized worklist instead of a wall of text.

The LLM (Claude) handles translation and quality assessment. The pipeline logic around it is what makes this a tool rather than a prompt.

---

## What We Have (Active)

| Asset | Location | Purpose in v3 |
|-------|----------|---------------|
| 500 HTER-rated segments | `data/hter_training/training_pairs.jsonl` | Eval benchmark + few-shot examples for LLM QE calibration |
| 53 glossary terms (medical) | `data/hter_training/glossary.tsv` | Injected into translation prompt |
| 3,684 extracted patent terms | `data/glosswerk_terminology.db` | Seed terminology for new domains |
| 200 HTER eval segments | `hter_evaluation.tsv` | Baseline comparison dataset |
| HTER Streamlit app | `scripts/hter_training_builder.py` | Already uses Claude translation with glossary injection — reusable |
| System prompt with IS rules | In hter_training_builder.py | Starting point for translation prompt |
| V2 pipeline output JSONs | Root `.json` files | Reference for output format |
| Test patents (DE .docx) | `mitralvalvestent.docx`, `test_patent.docx` | Test inputs for prototype |

## What We Archived (38GB)

All T5 checkpoints, LoRA adapters, OPUS training databases, T5 training scripts. Recoverable from `archive/` if ever needed. They won't be.

---

## The Pipeline (4 Components)

### Component 1: Source-Side Structural Analyzer

**Input:** German sentences
**Output:** Per-sentence structural risk profile

What it computes:
- Clause nesting depth (dependency parse)
- Verb-final span length (tokens between subject and clause-final verb)
- Relative clause chains
- Participial constructions requiring finite clause unpacking in English
- Sentence length (already proven to correlate with error rate)
- A composite "reordering difficulty" score

**Implementation:** spaCy German model (`de_core_news_lg`) for dependency parsing + rule-based feature extraction. No LLM needed. Fast, deterministic, scalable.

**Why this matters:** This is the piece that isn't prompt engineering. It produces a computed linguistic signal that the translator can't replicate by reading the prompt. It tells the LLM *where to pay attention* and tells the QE *where to be skeptical*.

### Component 2: LLM Translator

**Input:** Full German patent text + glossary + structural risk annotations
**Output:** English translation with per-segment metadata

Design:
- Claude receives the entire patent (or large sections) in one call for document-level consistency
- System prompt includes: patent conventions, information structure rules, glossary as mandatory terminology
- High-risk segments (from Component 1) get explicit inline instructions: "This sentence contains a 3-level nested subordinate clause — restructure for English front-loading"
- Output is structured (JSON) with one English sentence per German source sentence

**Key difference from "just prompting":** The structural analyzer's output is injected per-segment, so the translation prompt adapts to the specific difficulty of each sentence. A static prompt treats all sentences the same.

### Component 3: LLM Quality Estimator

**Input:** German source + English translation + structural risk score
**Output:** Per-segment rating (good/minor/major/critical) + error category + explanation

Design:
- Second Claude call evaluates the translation
- Few-shot calibrated using 30-50 selected examples from the 500 HTER-rated segments (stratified by rating)
- Structural risk score is provided as context: "This segment had high reordering difficulty — verify that clause order was restructured for English"
- Returns structured assessment: rating, error type (terminology / reordering / omission / other), brief explanation

**Baseline comparison:** Run XCOMET-XL in parallel on the same segments. Compare LLM QE ratings against XCOMET scores and against the 200 HTER ground-truth ratings. This tells us whether LLM QE is actually better or just different.

### Component 4: Document Assembly & Triage

**Input:** Translations + QE ratings + glossary
**Output:** Annotated .docx with color-coded segments + triage summary

- Terminology consistency pass (majority-vote enforcement + glossary override)
- Color coding: green (publish), orange (review), red (full edit)
- Triage summary with segment counts and flagged error types
- Export as .docx (for translators) and .json (for programmatic use)

---

## Build Order

**Phase 1: Structural Analyzer (build first)**
This is the novel component and the one that justifies the tool's existence. If the structural features don't actually predict translation difficulty better than sentence length alone, the whole thesis weakens. Build it, run it on the test patents, correlate against existing HTER ratings. This is a validation step before building the rest.

**Phase 2: LLM Translator (adapt existing)**
The HTER Streamlit app already does this. Extract the translation logic into a standalone module, add structural annotation injection, test on the two existing patents.

**Phase 3: LLM QE (new)**
Build the few-shot QE prompt, calibrate against the 500 HTER ratings, compare to XCOMET-XL baseline. This tells us whether LLM QE adds value over existing tools.

**Phase 4: Assembly & Demo**
Wire it together, produce annotated .docx output, run on a fresh patent end-to-end.

---

## Honest Assessment of Utility

**What makes this defensible:**
- The structural analyzer produces a signal that no current QE tool provides. XCOMET-XL flags errors after translation; this predicts difficulty before translation. That's a different and complementary capability.
- Document-level terminology consistency is a real, unsolved problem in commercial MT. DeepL doesn't do it. Google doesn't do it. Translators spend significant time on it manually.
- The triage output changes translator workflow. Instead of reading every segment, they skip green, skim orange, focus on red. That's measurable time savings.

**What's thin:**
- The LLM translation itself. Claude translating a patent is not a product. Any translator with an API key can do this. The translation is a commodity component.
- The LLM QE, on its own. If the few-shot calibration doesn't measurably outperform XCOMET-XL, it's not adding value — it's just more expensive.
- The glossary. 53 terms for one subdomain is a start, not a product. Scaling glossaries requires either translator input per project or automated extraction at scale.

**What would kill this:**
- If the structural analyzer doesn't predict translation difficulty better than a simple sentence-length threshold. Then it's just complexity for complexity's sake.
- If LLM translation + LLM QE produces results that are indistinguishable from "paste into Claude and ask it to translate." Then there's no tool, just a wrapper.
- If translators don't trust automated triage enough to skip green segments without checking. Then the time savings evaporate.

**What would validate this:**
- Structural risk scores correlate with HTER ratings at r > 0.5 (beyond what sentence length alone predicts).
- LLM QE matches human HTER ratings more closely than XCOMET-XL on the 200-segment eval set.
- End-to-end pipeline on a fresh patent produces >50% publishable segments with <5% false positives (segments rated "publish" that a translator would change).
- A translator who sees the demo says "I would use this."

---

## For the Research Paper

If the prototype validates, the paper gains a new section (or becomes a second paper):

- Section on information structure mismatch as a predictive feature for MT quality
- Empirical validation: structural features vs. HTER ratings
- Comparison: LLM QE vs. XCOMET-XL vs. human ratings
- End-to-end pipeline results on unseen patent

If it doesn't validate, the current paper stands on its own — the APE/QE findings and the distribution shift analysis are already publishable.
