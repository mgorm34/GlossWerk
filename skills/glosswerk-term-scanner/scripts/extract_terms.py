"""
GlossWerk Terminology Extractor

Extracts recurring German nouns, technical adjectives, and key verbs
from a patent .docx. Uses spaCy if available, falls back to
capitalization + suffix heuristics.

Usage:
    python extract_terms.py --input patent.docx --output candidates.json
    python extract_terms.py --input patent.docx --output candidates.json --min-freq 5
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


# Patent-specific abbreviations and tokens to skip
SKIP_TOKENS = {
    "FIG", "Fig", "Anspruch", "Ansprüche", "Abs", "Nr", "Bsp",
    "DE", "EP", "WO", "US", "PCT",
}

# Common German function words that happen to appear capitalized at sentence start
FUNCTION_WORDS = {
    "Der", "Die", "Das", "Ein", "Eine", "Eines", "Einem", "Einen", "Einer",
    "Dieser", "Diese", "Dieses", "Diesem", "Diesen",
    "Jeder", "Jede", "Jedes", "Jedem", "Jeden",
    "Welcher", "Welche", "Welches", "Welchem", "Welchen",
    "Alle", "Aller", "Allen", "Allem",
    "Einige", "Einiger", "Einigen", "Einigem",
    "Andere", "Anderer", "Anderen", "Anderem", "Anderes",
    "Mehrere", "Mehrerer", "Mehreren",
    "Solcher", "Solche", "Solches", "Solchem", "Solchen",
    "Dabei", "Dadurch", "Dafür", "Dagegen", "Daher", "Dahin",
    "Damit", "Danach", "Daneben", "Darin", "Darüber", "Darum",
    "Darunter", "Darauf", "Daraus", "Davon", "Davor", "Dazu",
    "Hierbei", "Hierdurch", "Hierfür", "Hierzu", "Hierin",
    "Ferner", "Weiterhin", "Außerdem", "Insbesondere", "Vorzugsweise",
    "Bevorzugt", "Gemäß", "Erfindungsgemäß", "Beispielsweise",
    "Dabei", "Somit", "Jedoch", "Zudem", "Wobei",
    "Wenn", "Weil", "Obwohl", "Während", "Nachdem", "Bevor",
    "Es", "Er", "Sie", "Wir", "Ihr", "Man",
    "Auch", "Nur", "Noch", "Schon", "Sehr", "Nicht",
    "Wie", "Was", "Wer", "Wo", "Wann", "Warum",
    "Oder", "Und", "Aber", "Denn", "Sondern",
    "Im", "In", "An", "Auf", "Aus", "Bei", "Mit", "Nach",
    "Von", "Vor", "Zu", "Über", "Unter", "Zwischen", "Durch", "Für",
}

# English function words that can appear in bilingual patents (EP abstracts, etc.)
# or if translated text is accidentally fed to the extractor
ENGLISH_FUNCTION_WORDS = {
    "The", "This", "That", "These", "Those", "Which", "Where", "When",
    "What", "Who", "How", "Each", "Every", "Some", "Any", "All",
    "Most", "Many", "Much", "Few", "Several", "Other", "Another",
    "Both", "Either", "Neither", "Such", "Same", "Own",
    "Here", "There", "Thus", "Then", "Also", "Only", "Just",
    "More", "Less", "Very", "Most", "Least", "Rather", "Quite",
    "However", "Therefore", "Furthermore", "Moreover", "Nevertheless",
    "According", "Although", "Because", "Since", "While", "During",
    "After", "Before", "Between", "Within", "Without", "Through",
    "Into", "From", "With", "For", "About", "Above", "Below",
    "Said", "Wherein", "Whereby", "Herein", "Thereof", "Therein",
    "Preferably", "Alternatively", "Specifically", "Generally",
}

FUNCTION_WORDS_LOWER = {w.lower() for w in FUNCTION_WORDS | ENGLISH_FUNCTION_WORDS}

# --- Technical adjective detection ---
# German technical adjectives with suffixes that signal domain-specific meaning.
# These are lowercase mid-sentence but critically important for translation
# because the literal translation is often wrong.
TECHNICAL_ADJ_SUFFIXES = (
    "bar",        # resorbierbar, dehnbar, implantierbar
    "förmig",     # schildförmig, ringförmig, rohrförmig
    "gemäß",      # erfindungsgemäß, patentgemäß
    "eigen",      # körpereigen, materialeigen
    "haltig",     # kohlenstoffhaltig, sauerstoffhaltig
    "beständig",  # temperaturbeständig, korrosionsbeständig
    "fähig",      # funktionsfähig, leistungsfähig
    "artig",      # netzartig, gitterartig, schleierartig
    "mäßig",      # erfindungsmäßig, zweckmäßig
    "seitig",     # innenseitig, außenseitig
    "wärtig",     # gegenwärtig, auswärtig
    "weise",      # schrittweise, stufenweise, abschnittsweise (NOT beispielsweise/vorzugsweise — those are adverbs)
    "immanent",   # materialimmanent
    "spezifisch", # domänenspezifisch, patientenspezifisch
    "abhängig",   # temperaturabhängig, druckabhängig
    "resistent",  # hitzeresistent, chemikalienresistent
    "kompatibel", # biokompatibel, MRT-kompatibel
    "hemmend",    # gerinnungshemmend, wachstumshemmend
    "tötend",     # keimtötend, virentötend
    "löslich",    # wasserlöslich, fettlöslich
    "durchlässig",# gasdurchlässig, lichtdurchlässig
    "wertig",     # hochwertig, minderwertig, gleichwertig
)

# Common German verbs that are NOT technical — skip these
# Adverbs that match technical suffixes but aren't technical adjectives
ADVERB_SKIP = {
    "beispielsweise", "vorzugsweise", "üblicherweise", "normalerweise",
    "idealerweise", "notwendigerweise", "vergleichsweise", "teilweise",
    "andererseits", "einerseits", "insbesondere", "möglicherweise",
}

COMMON_VERBS = {
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten", "kann", "können", "konnte", "konnten",
    "soll", "sollen", "sollte", "sollten", "muss", "müssen", "musste", "mussten",
    "darf", "dürfen", "durfte", "durften", "mag", "mögen", "mochte", "mochten",
    "sei", "seien", "wäre", "wären", "würde", "würden",
    "sein", "haben", "werden", "können", "müssen", "sollen", "dürfen",
    "dass", "wenn", "weil", "obwohl", "während", "nachdem", "bevor",
    "gibt", "geben", "gegeben", "geht", "gehen", "gegangen",
    "macht", "machen", "gemacht", "kommt", "kommen", "gekommen",
    "steht", "stehen", "gestanden", "liegt", "liegen", "gelegen",
    "zeigt", "zeigen", "gezeigt", "stellt", "stellen", "gestellt",
    "findet", "finden", "gefunden", "bringt", "bringen", "gebracht",
    "nimmt", "nehmen", "genommen", "lässt", "lassen", "gelassen",
    "hält", "halten", "gehalten", "bleibt", "bleiben", "geblieben",
    "führt", "führen", "geführt", "setzt", "setzen", "gesetzt",
    "bildet", "bilden", "gebildet",
}

# Patent verbs with technical significance — these get mistranslated often
PATENT_VERB_PATTERNS = {
    "aufweis":   "aufweisen — comprise / exhibit / feature",
    "umfass":    "umfassen — comprise / encompass / include",
    "anordn":    "anordnen — arrange / dispose / position",
    "vorsehen":  "vorsehen — provide / be configured to",
    "vorseh":    "vorsehen — provide / be configured to",
    "vorsieht":  "vorsehen — provide / be configured to",
    "vorgesehen":"vorsehen — provide / be configured to",
    "ausgebild": "ausbilden — form / configure / design",
    "ausbild":   "ausbilden — form / configure / design",
    "erstreckt": "erstrecken — extend / span",
    "erstreck":  "erstrecken — extend / span",
    "befestig":  "befestigen — attach / fasten / secure",
    "eingesetzt":"einsetzen — insert / deploy / use",
    "einsetz":   "einsetzen — insert / deploy / use",
    "aufnehm":   "aufnehmen — receive / accommodate / accept",
    "aufnimmt":  "aufnehmen — receive / accommodate / accept",
    "aufgenommen":"aufnehmen — receive / accommodate / accept",
    "gekennzeichnet":"kennzeichnen — characterize (patent claim language)",
    "kennzeichn":"kennzeichnen — characterize (patent claim language)",
    "angeordnet":"anordnen — arranged / disposed / positioned",
    "eingreif":  "eingreifen — engage / intervene / mesh",
    "eingreift": "eingreifen — engage / intervene / mesh",
    "zusammenwirk":"zusammenwirken — cooperate / interact / work together",
    "beaufschlag":"beaufschlagen — apply / subject to / act upon",
    "verrastet": "verrasten — latch / snap into place",
    "verrast":   "verrasten — latch / snap into place",
    "einrast":   "einrasten — engage / lock / snap in",
    "abdicht":   "abdichten — seal / make watertight",
    "abgedichtet":"abdichten — sealed",
}


def extract_text_from_docx(filepath):
    """Extract plain text from a .docx file."""
    if not HAS_DOCX:
        print("ERROR: python-docx required. Install: pip install python-docx")
        sys.exit(1)
    doc = DocxDocument(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def extract_sentences(paragraphs):
    """Split paragraphs into sentences, preserving patent conventions."""
    sentences = []
    abbrev_pattern = re.compile(
        r'\b(z\.B\.|bzw\.|ggf\.|d\.h\.|u\.a\.|o\.ä\.|etc\.|ca\.|vgl\.|Fig\.|FIG\.|Abs\.|Nr\.)\s*'
    )
    for para in paragraphs:
        para = re.sub(r'^\[\d{4}\]\s*', '', para)
        protected = abbrev_pattern.sub(lambda m: m.group(0).replace('.', '§DOT§'), para)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', protected)
        for part in parts:
            restored = part.replace('§DOT§', '.')
            if len(restored) > 10:
                sentences.append(restored)
    return sentences


def is_german_noun_candidate(token, position_in_sentence):
    """Check if a token looks like a German noun based on capitalization."""
    if not token or len(token) < 2:
        return False
    if not token[0].isupper():
        return False
    if token.isupper() and len(token) <= 4:
        return False
    if any(c.isdigit() for c in token):
        return False
    if token in FUNCTION_WORDS or token in SKIP_TOKENS:
        return False
    if token.lower() in FUNCTION_WORDS_LOWER:
        return False
    return True


def extract_nouns_heuristic(sentences):
    """Extract German nouns using capitalization heuristics."""
    mid_sentence_caps = Counter()
    initial_caps = Counter()
    all_caps = Counter()

    for sent in sentences:
        tokens = re.findall(r'\b[A-ZÄÖÜa-zäöüß][a-zäöüßA-ZÄÖÜ-]+\b', sent)
        for i, token in enumerate(tokens):
            if not is_german_noun_candidate(token, i):
                continue
            all_caps[token] += 1
            if i == 0:
                initial_caps[token] += 1
            else:
                mid_sentence_caps[token] += 1

    nouns = {}
    for token, count in all_caps.items():
        mid_count = mid_sentence_caps.get(token, 0)
        if mid_count >= 1:
            # Appeared capitalized mid-sentence at least once → likely a German noun
            nouns[token] = count
        elif count >= 8 and initial_caps.get(token, 0) >= 5:
            # Only appeared at sentence start, but very frequently.
            # Higher threshold to prevent English words leaking through
            # when processing bilingual or accidentally-English text.
            nouns[token] = count

    return nouns


def extract_nouns_spacy(sentences):
    """Extract German nouns using spaCy dependency parsing.

    Tracks POS tags per token across all sentences and only includes words
    that are tagged as NOUN/PROPN more than half the time. This prevents
    words like 'Am', 'So', 'Wesentlichen' that spaCy inconsistently tags
    from leaking into the noun list.
    """
    nlp = spacy.load("de_core_news_lg")
    # Track noun vs non-noun tags per surface form
    noun_tag_counts = Counter()  # times tagged as NOUN/PROPN
    total_tag_counts = Counter()  # total occurrences
    lemma_map = defaultdict(set)

    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            if len(token.text) < 2:
                continue
            if token.text in SKIP_TOKENS:
                continue
            if token.text.lower() in FUNCTION_WORDS_LOWER:
                continue
            # Count all capitalized tokens
            if token.text[0].isupper():
                total_tag_counts[token.text] += 1
                if token.pos_ in ("NOUN", "PROPN"):
                    noun_tag_counts[token.text] += 1
                    lemma_map[token.lemma_].add(token.text)

        # Extract noun compounds (adjacent nouns in noun chunks)
        for chunk in doc.noun_chunks:
            compound_parts = []
            for t in chunk:
                if t.pos_ in ("NOUN", "PROPN") and t.text[0].isupper():
                    compound_parts.append(t.text)
            if len(compound_parts) >= 2:
                compound = " ".join(compound_parts)
                noun_tag_counts[compound] += 1
                total_tag_counts[compound] += 1

    # Only include words tagged as NOUN/PROPN more than half the time.
    # This filters out words like 'Am', 'So' that spaCy sometimes
    # mistags as nouns in certain sentence positions.
    noun_counts = {}
    for token, noun_count in noun_tag_counts.items():
        total = total_tag_counts.get(token, noun_count)
        if noun_count > total * 0.5:
            noun_counts[token] = noun_count

    # Filter out frozen expressions that look like nouns.
    # "Im Wesentlichen" = "essentially", "Im Folgenden" = "in the following"
    # These are adverbial phrases where a nominalized adjective appears
    # capitalized but shouldn't be in a terminology glossary.
    FROZEN_EXPRESSIONS = {
        "Wesentlichen", "Folgenden", "Allgemeinen", "Besonderen",
        "Einzelnen", "Übrigen", "Weiteren", "Vorstehenden",
        "Nachstehenden", "Obigen", "Untenstehenden",
    }
    for frozen in FROZEN_EXPRESSIONS:
        noun_counts.pop(frozen, None)

    return dict(noun_counts), dict(lemma_map)


def extract_technical_adjectives(sentences, min_freq=2):
    """
    Extract recurring technical adjectives from German patent text.

    German technical adjectives are lowercase mid-sentence, so noun extraction
    misses them. We detect them by suffix patterns that indicate domain-specific
    meaning (e.g., -bar, -förmig, -gemäß, -eigen).

    These matter because the literal translation is often wrong:
      körpereigene → "endogenous" not "body's own"
      resorbierbar → "resorbable" not "absorbable"
      erfindungsgemäß → "according to the invention" not "inventive"
    """
    adj_counts = Counter()
    all_tokens = Counter()

    for sent in sentences:
        # Tokenize — grab lowercase words too
        tokens = re.findall(r'\b[a-zäöüß][a-zäöüß-]+\b', sent.lower())
        for token in tokens:
            all_tokens[token] += 1

    # Filter to words with technical suffixes.
    # German adjectives inflect: -e, -en, -er, -es, -em, -em
    # So "erfindungsgemäße" needs to be recognized as "erfindungsgemäß" + -e
    INFLECTION_ENDINGS = ("en", "em", "er", "es", "e")

    for token, count in all_tokens.items():
        if count < min_freq:
            continue
        if len(token) < 6:
            continue
        if token in COMMON_VERBS or token in ADVERB_SKIP:
            continue

        # Try direct suffix match first, then strip inflection and retry
        matched = False
        for suffix in TECHNICAL_ADJ_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                adj_counts[token] = count
                matched = True
                break

        if not matched:
            # Strip German adjective inflection endings and retry
            stripped = token
            for ending in INFLECTION_ENDINGS:
                if token.endswith(ending) and len(token) > len(ending) + 4:
                    stripped = token[:-len(ending)]
                    break
            if stripped != token:
                for suffix in TECHNICAL_ADJ_SUFFIXES:
                    if stripped.endswith(suffix) and len(stripped) > len(suffix) + 2:
                        adj_counts[token] = count
                        break

    # Cluster inflected forms: erfindungsgemäße + erfindungsgemäßen → erfindungsgemäß
    clustered = defaultdict(lambda: {"frequency": 0, "variants": []})
    for token, count in adj_counts.items():
        # Find the base form by stripping inflection
        base = token
        for ending in INFLECTION_ENDINGS:
            if token.endswith(ending) and len(token) > len(ending) + 4:
                base = token[:-len(ending)]
                break
        clustered[base]["frequency"] += count
        if token != base:
            clustered[base]["variants"].append(token)

    return {k: v["frequency"] for k, v in clustered.items()}, {
        k: v["variants"] for k, v in clustered.items()
    }


def extract_patent_verbs(sentences, min_freq=3):
    """
    Extract patent-specific verbs that commonly get mistranslated.

    Unlike nouns and adjectives, verbs should NOT go in the enforced glossary
    because the same verb often needs different translations in different
    syntactic contexts (claims vs description, active vs passive).

    Instead, these are flagged as a reference for the translator with
    context-dependent usage notes.
    """
    verb_hits = Counter()
    verb_info = {}

    full_text = " ".join(sentences).lower()
    tokens = re.findall(r'\b[a-zäöüß]+\b', full_text)
    token_counts = Counter(tokens)

    for stem, info in PATENT_VERB_PATTERNS.items():
        # Count all tokens that contain this stem
        count = 0
        matched_forms = set()
        for token, tc in token_counts.items():
            if stem in token and token not in COMMON_VERBS:
                count += tc
                matched_forms.add(token)

        if count >= min_freq and matched_forms:
            # Use the base form from our info string
            base_form = info.split(" — ")[0].strip()
            if base_form not in verb_info or count > verb_hits.get(base_form, 0):
                verb_hits[base_form] = count
                verb_info[base_form] = {
                    "frequency": count,
                    "translations": info.split(" — ")[1].strip() if " — " in info else "",
                    "forms_found": sorted(matched_forms)[:5],  # cap at 5 forms
                }

    return verb_info


def cluster_variants(nouns):
    """Group surface form variants (e.g., Katheter/Katheters/Kathetern)."""
    clusters = defaultdict(list)
    sorted_nouns = sorted(nouns.keys(), key=len)

    assigned = set()
    for noun in sorted_nouns:
        if noun in assigned:
            continue
        base = noun.rstrip('s').rstrip('n').rstrip('e')
        if len(base) < 3:
            base = noun
        cluster_members = [(noun, nouns[noun])]

        for other in sorted_nouns:
            if other == noun or other in assigned:
                continue
            other_base = other.rstrip('s').rstrip('n').rstrip('e')
            if len(other_base) < 3:
                other_base = other
            if (base == other_base or
                other.startswith(noun) and len(other) - len(noun) <= 3 or
                noun.startswith(other) and len(noun) - len(other) <= 3):
                cluster_members.append((other, nouns[other]))
                assigned.add(other)

        cluster_members.sort(key=lambda x: -x[1])
        primary = cluster_members[0][0]
        total_freq = sum(c for _, c in cluster_members)
        variants = [m[0] for m in cluster_members if m[0] != primary]

        clusters[primary] = {
            "frequency": total_freq,
            "variants": variants
        }
        assigned.add(noun)

    for noun in sorted_nouns:
        if noun not in assigned:
            clusters[noun] = {
                "frequency": nouns[noun],
                "variants": []
            }

    return dict(clusters)


def main():
    parser = argparse.ArgumentParser(description="Extract German patent terminology")
    parser.add_argument("--input", required=True, help="German patent .docx file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="Minimum noun frequency to include (default: 3)")
    parser.add_argument("--adj-min-freq", type=int, default=2,
                        help="Minimum adjective frequency to include (default: 2)")
    parser.add_argument("--verb-min-freq", type=int, default=3,
                        help="Minimum verb frequency to include (default: 3)")
    parser.add_argument("--use-spacy", action="store_true",
                        help="Force spaCy mode (requires de_core_news_lg)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    print(f"Extracting text from {args.input}...")
    paragraphs = extract_text_from_docx(args.input)
    sentences = extract_sentences(paragraphs)
    print(f"  Found {len(paragraphs)} paragraphs, {len(sentences)} sentences")

    # --- Nouns ---
    lemma_map = {}
    if args.use_spacy or (HAS_SPACY and not args.use_spacy):
        try:
            print("Using spaCy for noun extraction...")
            nouns, lemma_map = extract_nouns_spacy(sentences)
        except OSError:
            print("  spaCy model not found, falling back to heuristic extraction")
            nouns = extract_nouns_heuristic(sentences)
    else:
        print("Using heuristic noun extraction (install spaCy for better results)")
        nouns = extract_nouns_heuristic(sentences)

    filtered_nouns = {k: v for k, v in nouns.items() if v >= args.min_freq}
    clusters = cluster_variants(filtered_nouns)
    sorted_nouns = sorted(clusters.items(), key=lambda x: -x[1]["frequency"])
    print(f"  Nouns: {len(nouns)} unique, {len(sorted_nouns)} with freq >= {args.min_freq}")

    # --- Technical adjectives ---
    print("Extracting technical adjectives...")
    adjectives, adj_variants = extract_technical_adjectives(sentences, min_freq=args.adj_min_freq)
    sorted_adjs = sorted(adjectives.items(), key=lambda x: -x[1])
    print(f"  Adjectives: {len(sorted_adjs)} technical adjectives found")

    # --- Patent verbs ---
    print("Extracting patent-specific verbs...")
    verbs = extract_patent_verbs(sentences, min_freq=args.verb_min_freq)
    sorted_verbs = sorted(verbs.items(), key=lambda x: -x[1]["frequency"])
    print(f"  Verbs: {len(sorted_verbs)} patent verbs flagged")

    # --- Build output ---
    output = {
        "source_file": os.path.basename(args.input),
        "total_sentences": len(sentences),
        "extraction_method": "spacy" if lemma_map else "heuristic",
        "nouns": {
            "total": len(sorted_nouns),
            "min_frequency": args.min_freq,
            "terms": [
                {
                    "de": term,
                    "frequency": data["frequency"],
                    "variants": data["variants"],
                }
                for term, data in sorted_nouns
            ]
        },
        "adjectives": {
            "total": len(sorted_adjs),
            "min_frequency": args.adj_min_freq,
            "note": "Technical adjectives — enforce in glossary like nouns",
            "terms": [
                {
                    "de": adj,
                    "frequency": freq,
                    "variants": adj_variants.get(adj, []),
                }
                for adj, freq in sorted_adjs
            ]
        },
        "verbs": {
            "total": len(sorted_verbs),
            "min_frequency": args.verb_min_freq,
            "note": "Reference only — do NOT enforce. Context-dependent translation.",
            "terms": [
                {
                    "de": verb,
                    "frequency": info["frequency"],
                    "suggested_translations": info["translations"],
                    "forms_found": info["forms_found"],
                }
                for verb, info in sorted_verbs
            ]
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output}")

    print(f"\n{'='*60}")
    print(f"NOUNS ({len(sorted_nouns)} terms) — enforce in glossary")
    print(f"{'='*60}")
    for term, data in sorted_nouns[:15]:
        variants = f" (also: {', '.join(data['variants'])})" if data['variants'] else ""
        print(f"  {term} ({data['frequency']}x){variants}")
    if len(sorted_nouns) > 15:
        print(f"  ... and {len(sorted_nouns) - 15} more")

    print(f"\n{'='*60}")
    print(f"TECHNICAL ADJECTIVES ({len(sorted_adjs)} terms) — enforce in glossary")
    print(f"{'='*60}")
    for adj, freq in sorted_adjs[:15]:
        print(f"  {adj} ({freq}x)")
    if len(sorted_adjs) > 15:
        print(f"  ... and {len(sorted_adjs) - 15} more")

    print(f"\n{'='*60}")
    print(f"PATENT VERBS ({len(sorted_verbs)} terms) — reference only, not enforced")
    print(f"{'='*60}")
    for verb, info in sorted_verbs[:10]:
        print(f"  {verb} ({info['frequency']}x) → {info['translations']}")
        print(f"    forms: {', '.join(info['forms_found'])}")


if __name__ == "__main__":
    main()
