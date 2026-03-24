---
name: glosswerk-structural-analyzer
description: >
  Analyzes German patent sentences for structural features that predict
  DE→EN translation difficulty. Computes clause nesting depth, verb-final
  spans, relative clause chains, participial constructions, Vorfeld length,
  and a composite reordering difficulty score. Use before translation to
  identify high-risk segments that need restructuring instructions. This is
  always run after terminology scanning and before LLM translation.
---

# GlossWerk Source-Side Structural Analyzer

## What this does

Parses every sentence in a German patent to compute structural features that
predict how difficult the sentence will be to translate into English. The core
insight: German and English have fundamentally different information structure.
German builds toward the end (end-weight, end-focus, verb-final subordinate
clauses). English front-loads. The constructions that cause this mismatch are
detectable from the German source alone, before any translation happens.

The analyzer produces two things:
1. A **risk score** (0–1) per sentence that predicts reordering difficulty
2. **Translation hints** for medium/high-risk sentences — specific instructions
   injected into the LLM translation prompt

## Structural features computed

| Feature | What it detects | Why it matters for DE→EN |
|---------|----------------|--------------------------|
| Clause nesting depth | Stacked subordinate clauses (dass...weil...wenn) | Deep nesting must be unpacked; English can't mirror German clause stacking |
| Verb-final span | Distance from conjunction to clause-final verb | The longer the span, the more information arrives "too late" for English structure |
| Relative clause chains | Multiple stacked relative clauses (der...die...welcher) | Nested which/that chains are unreadable in English — need restructuring |
| Participial constructions | Extended pre-nominal participles (die im Herzen angeordnete Klappe) | Must be converted to post-nominal relative clauses in English |
| Vorfeld length | Tokens before the finite verb in main clauses | Long topicalized pre-fields need repositioning for English information flow |
| Sentence length | Token count | Baseline difficulty predictor — longer sentences compound all other risk factors |

## Risk score composition

The composite score (0–1) weights features by their observed impact on translation quality:

- Sentence length: up to 0.25
- Clause nesting depth: up to 0.25
- Verb-final span: up to 0.20
- Relative clauses: up to 0.10
- Participial constructions: up to 0.10
- Vorfeld length: up to 0.10

Thresholds: **low** (<0.25), **medium** (0.25–0.5), **high** (≥0.5)

## Workflow

### Step 1: Run the structural analysis

```bash
python skills/glosswerk-structural-analyzer/scripts/analyze_structure.py \
    --input <german.docx> \
    --output <analysis.json>
```

Optional: `--format tsv` for a quick spreadsheet-friendly overview.

### Step 2: Review the output

The JSON output contains per-sentence analysis:

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "Der in dem Herzen des Patienten angeordnete...",
      "features": {
        "n_tokens": 42,
        "clause_depth": 2,
        "max_verb_final_span": 8,
        "n_relative_clauses": 1,
        "n_participial_constructions": 1,
        "max_participial_span": 6,
        "vorfeld_length": 8,
        "risk_score": 0.45,
        "risk_level": "medium",
        "risk_factors": ["clause depth 2", "verb-final span of 8 tokens", "1 participial construction(s)", "long Vorfeld (8 tokens)"]
      },
      "translation_hint": "This sentence has 2-level clause nesting. Restructure for English..."
    }
  ],
  "summary": {
    "total_sentences": 137,
    "avg_risk_score": 0.18,
    "risk_distribution": {"high": 8, "medium": 31, "low": 98},
    "high_risk_pct": 5.8
  }
}
```

### Step 3: Feed into LLM translation

For medium/high-risk sentences, the `translation_hint` field contains
specific restructuring instructions to inject into the translation prompt.
The LLM translator skill reads these hints and includes them inline
when requesting translation.

Low-risk sentences get the standard translation prompt without extra instructions.

## Integration with the pipeline

```
[German .docx]
    │
    ├── Term Scanner (extract terminology)
    │
    ├── Structural Analyzer (this skill)
    │       ↓
    │   Per-sentence risk scores + translation hints
    │
    └── LLM Translator (receives glossary + structural hints)
            ↓
        LLM QE (receives risk scores as skepticism guide)
            ↓
        Document Assembly (color-coded output)
```

## Dependencies

- spaCy + de_core_news_lg — **required** (no heuristic fallback for this component)
- python-docx — required for .docx text extraction

Install: `pip install spacy python-docx && python -m spacy download de_core_news_lg`

## Notes on German structural linguistics

**Behaghel's laws** (1932): German follows the "law of increasing constituents"
(Gesetz der wachsenden Glieder) — heavier, more informative material goes to
the end of the sentence. English does the opposite. This fundamental mismatch
is why constituent reordering is the hardest part of DE→EN patent translation.

**Verb-second (V2) constraint**: In German main clauses, the finite verb must
be in second position. Everything before it (the Vorfeld) is topicalized.
In subordinate clauses, the verb goes to the very end (verb-final, VF).
Both of these create information ordering problems for English.

**Participial attributes**: German allows pre-nominal participial phrases of
arbitrary length: "die für den Einsatz im menschlichen Körper vorgesehene
selbstexpandierende Herzklappenprothese." English cannot do this — it must
be restructured as a post-nominal relative clause or broken into multiple
sentences.
