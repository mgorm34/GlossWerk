---
name: glosswerk-term-scanner
description: >
  Scans a German patent document to extract recurring technical terminology
  (nouns, technical adjectives, patent verbs) and propose English translation
  options for translator review. This is always the first step before translating
  a German patent. Use whenever: processing a new German patent, building a
  glossary, extracting terminology from German text, preparing for patent
  translation, or the user mentions terms, terminology, glossary, or Fachbegriffe.
  Also trigger when the user uploads a German .docx and mentions translation —
  terminology scanning should happen before translation.
---

# GlossWerk Terminology Scanner

## What this does

Before translating a German patent, you need to know what terms appear and how
the translator wants them rendered in English. This skill extracts three
categories of terminology from a patent document:

1. **Nouns & noun compounds** — the core glossary. Enforced consistently
   throughout translation. (e.g., Stentbügel → stent strut)
2. **Technical adjectives** — domain-specific adjectives where the literal
   translation is wrong. Also enforced. (e.g., körpereigene → endogenous,
   NOT "body's own"; faltbar → collapsible, NOT "foldable")
3. **Patent verbs** — verbs with specific patent meanings that vary by context.
   Flagged as reference for the translator, NOT enforced in glossary.
   (e.g., aufweisen → comprise / exhibit / feature, depending on context)

The distinction matters: nouns and technical adjectives get one consistent
translation throughout the document. Verbs don't, because the same verb
legitimately needs different English depending on syntactic context.

## Workflow

### Step 1: Extract candidate terms

Run the extraction script on the German .docx:

```bash
python skills/glosswerk-term-scanner/scripts/extract_terms.py \
    --input <german.docx> \
    --output <candidates.json>
```

The script extracts all three categories:

**Nouns** — identified by German capitalization rules (all German nouns are
capitalized mid-sentence). Clusters inflected variants (Katheter/Katheters/
Kathetern). If spaCy with `de_core_news_lg` is available, also detects
multi-word noun compounds via dependency parsing.

**Technical adjectives** — identified by suffix patterns that signal
domain-specific meaning: -bar (resorbierbar), -förmig (ringförmig),
-gemäß (erfindungsgemäß), -eigen (körpereigen), -kompatibel (biokompatibel),
-hemmend (gerinnungshemmend), etc. Clusters inflected forms
(erfindungsgemäße + erfindungsgemäßen → erfindungsgemäß, 16x total).
Filters out adverbs that share suffixes (beispielsweise, vorzugsweise).

**Patent verbs** — matched against a curated list of verbs that frequently
get mistranslated in patent MT: aufweisen, umfassen, anordnen, vorsehen,
ausbilden, kennzeichnen, befestigen, etc. Returns the base form, frequency,
observed inflected forms, and suggested translations with usage notes.

### Step 2: Propose English translations (LLM call)

Take the extracted terms and propose English translations. Use separate
prompts for each category:

**For nouns — system prompt:**
```
You are a DE→EN patent terminology expert. For each German noun or compound,
propose 2-3 English translations used in patent literature. Rank by convention —
the most standard patent translation first. If a term has one clearly dominant
translation in patents, just give that one.

For compound nouns, translate the full compound — do not break it apart.
For terms with a clear technical meaning, prefer the technical term over
a literal translation.
```

**For technical adjectives — system prompt:**
```
You are a DE→EN patent terminology expert. For each German technical adjective,
propose the correct English patent translation. These adjectives often have
a literal translation that is WRONG in patent context. Provide:
- The correct patent translation (first)
- The literal translation to AVOID (marked as such)
- A brief note if the distinction matters for patent validity

Example:
  körpereigene → "endogenous" (AVOID: "body's own")
  erfindungsgemäß → "according to the invention" (AVOID: "inventive")
  resorbierbar → "resorbable" (AVOID: "absorbable")
```

**For patent verbs** — no LLM call needed. The extraction script already
provides suggested translations from a curated list. Present these directly.

### Step 3: Present to translator

Format the results for translator review with three clearly separated sections:

```
TERMINOLOGY REVIEW — [patent filename]
Found [N nouns], [M adjectives], [K verbs] across [S] sentences.

═══════════════════════════════════════
SECTION 1: NOUNS — enforced in glossary
═══════════════════════════════════════

HIGH FREQUENCY (10+ occurrences):
  Stent (58x) → stent
  Stentbügel (29x) → stent strut | stent frame
  Mitralklappenstent (23x) → mitral valve stent

MEDIUM FREQUENCY (5-9 occurrences):
  Annulus (8x) → annulus
  Implantation (8x) → implantation
  ...

LOWER FREQUENCY (3-4 occurrences):
  Formgedächtnislegierung (4x) → shape memory alloy
  ...

═══════════════════════════════════════════════
SECTION 2: TECHNICAL ADJECTIVES — enforced in glossary
═══════════════════════════════════════════════

  erfindungsgemäß (16x) → according to the invention (AVOID: inventive)
  faltbar (4x) → collapsible (AVOID: foldable)
  selbstexpandierbar (2x) → self-expanding (AVOID: self-expandable)
  ...

═══════════════════════════════════════════════════════════
SECTION 3: PATENT VERBS — reference only, NOT in glossary
═══════════════════════════════════════════════════════════

  anordnen (11x) → arrange / dispose / position
    Context: "angeordnet" in claims = "disposed"; in description = "arranged"
  aufweisen (8x) → comprise / exhibit / feature
    Context: "aufweisen" in claims = "comprise"; elsewhere = "exhibit"
  kennzeichnen (10x) → characterize
    Context: "dadurch gekennzeichnet, dass" = "characterized in that"
  ...
```

For nouns and adjectives, the first option is the recommended default.
The translator confirms, picks an alternative, or types their own.
For verbs, the translator reads the context notes — no selection needed.

### Step 4: Export glossary

Save the translator's selections as a TSV glossary file. Include both
nouns AND technical adjectives. Do NOT include verbs.

```
de_term\ten_term
Stent\tstent
Stentbügel\tstent strut
erfindungsgemäß\taccording to the invention
faltbar\tcollapsible
```

Format: tab-separated, no header row, `de_term\ten_term`

This file is the input for the translation skill's glossary parameter.

## Output contract

The skill produces three files:
1. `<patent_name>_candidates.json` — raw extraction with all three categories
2. `<patent_name>_glossary.tsv` — final glossary (nouns + adjectives only)
3. `<patent_name>_verb_reference.md` — verb reference sheet for translator

## Dependencies

- python-docx (`pip install python-docx`) — required for .docx text extraction
- spaCy + de_core_news_lg — optional, improves compound noun detection
- If spaCy unavailable, falls back to heuristic extraction (reliable for German)

## Notes on extraction methods

**Nouns:** German capitalizes all nouns, making extraction reliable without NLP.
Edge cases handled: sentence-initial words (checked for mid-sentence occurrence),
nominalized verbs (captured correctly since they're capitalized in German).

**Technical adjectives:** Detected by suffix patterns (-bar, -förmig, -gemäß,
-eigen, -kompatibel, -hemmend, etc.) with German adjective inflection stripping
(-e, -en, -er, -es, -em). Adverbs with matching suffixes (beispielsweise,
vorzugsweise) are filtered out.

**Patent verbs:** Matched against a curated dictionary of ~25 patent-specific
verbs with known mistranslation patterns. Not exhaustive — the translator may
flag additional verbs during review.
