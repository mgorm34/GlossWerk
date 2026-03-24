# GlossWerk Terminology Scan Summary

## Document Processed
- **File**: mitralvalvestent.docx
- **Type**: German patent (mitral valve stent)
- **Processing Date**: 2026-03-22

## Extraction Results

### Extraction Details
- **Total paragraphs**: 114
- **Total sentences**: 137
- **Extraction method**: Heuristic (capitalization-based German noun detection)
- **Unique nouns found**: 314
- **Terms meeting frequency threshold (≥3)**: 54

### Frequency Distribution
- **High frequency (18+ occurrences)**: 4 terms
- **Medium-high frequency (15-17 occurrences)**: 3 terms
- **Medium frequency (10-14 occurrences)**: 6 terms
- **Medium-low frequency (6-9 occurrences)**: 10 terms
- **Lower frequency (4-5 occurrences)**: 11 terms
- **Lowest frequency (3 occurrences)**: 20 terms

## Key Terminology Findings

### Most Frequent Terms
1. **Stent** (58x) - Core device term
2. **Stentbügel** (29x) - Stent frame/structure
3. **Mitralklappenstent** (23x) - Primary device type
4. **Abschnitt** (18x) - Document sectioning

### Anatomical Terms
The document contains standard cardiac anatomy terminology:
- Mitralklappe, Aortenklappe (valve types)
- Vorhof, Ventrikel (heart chambers)
- Annulus, Chordae (valve structures)

### Device-Specific Terms
- Stentbügel (stent frame)
- Formgedächtnislegierung (shape memory alloy)
- Kartusche (delivery cartridge)
- Verankerungselementen (anchoring elements)

### Figure/Drawing References
Multiple view-related terms identified:
- Ansicht, Darstellung (views)
- Draufsicht, Seitenansicht, Lateral-Ansicht (specific view types)
- Detailansicht (detail view)

## Deliverables Generated

### 1. mitralvalvestent_candidates.json
Raw extraction output from the extraction script containing:
- All 54 terms with frequencies
- Variant forms (inflected plurals, genitive cases, etc.)
- Frequency counts grouped by primary form

### 2. term_review.md
Formatted terminology review document organized by frequency bands:
- 54 terms with translation options (2-3 per term)
- Medical/anatomical notes explaining context-dependent choices
- Instructions for translator review and selection
- Ready for translator approval workflow

### 3. glossary.tsv
Tab-separated glossary file containing:
- German term (left column)
- Recommended English translation (right column)
- 54 term pairs formatted for downstream translation tools
- No header row (clean for import)

### 4. translations.json
Complete translation proposal with:
- All 54 German terms
- 2-3 English options per term (ranked by convention)
- Contextual notes explaining disambiguation where needed
- Medical device/patent terminology standards applied

## Translation Quality Notes

### Terminology Consistency Opportunities
1. **Valve terminology**: "Klappe" appears in multiple compounds - using unified "valve" base ensures consistency
2. **View references**: 5 different view-related terms all translate to "view" - standardizing on this term avoids variation
3. **Anchoring terms**: Both "Verankerung" and "Befestigung" relate to attachment - recommended distinctions provided
4. **Plural forms**: "Mehrzahl" and "Vielzahl" both express multiplicity - context-appropriate options provided

### High-Risk Terms (Review Recommended)
- **Kartusche** in medical context - confirmed as "cartridge" in device delivery systems
- **Stützbügel** vs **Stentbügel** - both frame types but distinct structures
- **Darstellung** - requires "view" for figures, "representation" for technical description
- **Zustand** - context determines whether "state," "condition," or "configuration"

## Next Steps

1. **Translator Review**: Open term_review.md with translator
2. **Selection**: Translator confirms recommended options or selects alternatives
3. **Glossary Export**: Approved selections exported to final glossary.tsv
4. **Translation**: Use glossary.tsv as input to translation workflow for consistency

## Files Location
All outputs saved to:
```
/sessions/admiring-relaxed-mccarthy/mnt/glosswerk/skills/glosswerk-term-scanner-workspace/iteration-1/eval-1-mitral-valve/with_skill/outputs/
```
