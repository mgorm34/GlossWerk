# GlossWerk — Full Project Handoff Document

**Date:** March 24, 2026
**Author:** Matt Gorman (mjg@glosswerk.com)
**Purpose:** Comprehensive context for resuming development in a new Claude session

---

## 1. What GlossWerk Is

GlossWerk is a **DE→EN patent translation pipeline** with built-in quality estimation, terminology management, and structural analysis. It's built as a **Streamlit web app** targeting **Language Service Providers (LSPs)** who handle high-volume German patent translation.

The core value proposition: instead of a human reviewer reading every translated segment, GlossWerk triages each segment as green (publishable), orange (quick review), or red (needs editing) — so reviewers only spend time where it matters.

## 2. Architecture & File Map

### Core Pipeline Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `glosswerk_app.py` | **Main Streamlit app** — the integrated UI with tabs for Upload, Nouns, Verbs, Translate, Review, Export |
| `translate.py` | LLM translator (Claude API) with system prompt for patent-specific DE→EN translation. Includes rules for lexical redundancy avoidance and complex sentence restructuring. |
| `quality_estimate.py` | QE evaluator (Claude API) that rates each segment good/minor/major/critical. Includes rules for calque detection, readability/information overload, lexical redundancy from German morphology, and multi-failure escalation. |
| `demo_auth.py` | Demo authentication & usage gating. Generates invite codes (GW-XXXXXXXX), validates them, tracks patent usage (3 patent limit, 14-day expiry). CLI: `python demo_auth.py generate --company "X" --email "y@z.com"` |
| `assemble.py` | Document assembly — takes translated segments and builds final .docx output |
| `term_scanner_app.py` | Terminology extraction — scans patent for domain-specific nouns/verbs, generates glossary |

### Other Scripts (legacy/experimental)
- `05_evaluate.py` through `14_process_bigquery_data.py` — earlier pipeline versions, HTER evaluation tools, BigQuery data processing. Not part of the current app.
- `hter_eval_app.py` / `hter_training_builder.py` — HTER-based evaluation tools from earlier development phase.

### Website (`website/`)
- `index.html` — Single-file landing page at glosswerk.com. DM Sans font, dark gradient hero, emerald green (#10b981) accents. Sections: Hero, How It Works (3 steps), Key Features (5 cards), Quality Triage demo, For LSPs (stats), Request Demo form, Footer.

### Business (`business/`)
- `glosswerk_business_plan.docx` — Full business plan. Pricing: Phase 1 free demo → Phase 2 $75-150/patent → Phase 3 $0.03-0.05/word or subscription ($500-$1500/month).

### Data (`data/`)
- `demo_users.json` — Usage tracking per invite code
- `demo_codes.json` — Generated invite codes with company/email metadata

### Config
- `.gitignore` — Comprehensive ignores for Python, temp files, test outputs
- `requirements.txt` — Python dependencies
- `DEPLOY.md` — Deployment guide (Vercel + Railway)
- `PLAN.md` — Project plan

## 3. Key Technical Details

### QE Prompt Rules
The quality_estimate.py system prompt includes these specialized rules:
1. Standard translation quality criteria (accuracy, fluency, terminology)
2. Calque detection (5 specific German→English calque patterns)
3. Multi-failure escalation: 2+ distinct errors → at least "major"; already-major + second error → "critical"
4. Readability/information overload: deeply nested modifiers that force re-reading → "minor" minimum
5. Lexical redundancy from German morphology: related German words (Notfall + Nottracheotomie) mapping to same English root
6. Suggestions must: be complete sentences, fix ALL issues, wrap changed portions in `**bold**` markers

### Translation Prompt Rules
The translate.py system prompt includes:
- Patent-specific translation conventions
- Lexical redundancy avoidance section (German morphological relatives)
- Complex sentence restructuring section (nested prenominal participial phrases)

### Streamlit App Features
- **Upload tab**: Upload German patent .docx, configure term scanning
- **Nouns tab**: Review extracted nouns with frequency, select translations, add manual entries
- **Verbs tab**: Review verbs with tag-based translation system (add/remove individual translations)
- **Translate tab**: Run LLM translation with glossary enforcement
- **Review tab**: Side-by-side QE review with color-coded segments, one-click "Apply Fix" (doesn't lock segment), bold highlighting on changed portions, progress tracker showing % confirmed
- **Export tab**: Generate final .docx with watermark in demo mode
- **Adaptive noun frequency threshold**: `max(2, min(8, round(n_sentences / 20)))` — scales with patent length
- **Demo mode**: Toggled by `GLOSSWERK_DEMO=true` env var. Shows auth gate, patent counter, day counter, hides API key/model settings, adds watermark to exports.

### Form & Email
- Demo request form on glosswerk.com POSTs to Formspree (ID: `xwvrgjye`)
- Form fields: Full Name, Company, Email Address, Role (dropdown: PM, Translator/Post-Editor, QM, Executive/Director, Other)
- Formspree notifications need to be configured to send to mjg@glosswerk.com (currently goes to Gmail)

## 4. Infrastructure Status

### Live & Working
- **glosswerk.com** → Vercel (static HTML landing page) ✅
- **Domain**: glosswerk.com on Namecheap, nameservers pointed to Vercel (ns1/ns2.vercel-dns.com) ✅
- **Email**: mjg@glosswerk.com on Zoho Mail Lite ✅
  - MX records: mx.zoho.com (10), mx2.zoho.com (20), mx3.zoho.com (50)
  - SPF and DKIM TXT records configured
- **Form**: Formspree endpoint working, submissions visible in dashboard ✅
- **GitHub**: github.com/mgorm34/GlossWerk (branch: `main`) ✅
- **SSL**: Auto-provisioned by Vercel ✅

### NOT Yet Set Up
- **Railway** (app.glosswerk.com) — The Streamlit app is NOT deployed yet. It only runs locally. This is the critical gap: clients can request a demo on the website, but they can't actually use the tool until Railway is set up OR you share via ngrok/Cloudflare Tunnel.
- **Formspree → Zoho notification**: Formspree currently sends to Gmail (signup email). Need to change notification email to mjg@glosswerk.com in Formspree settings.

### Deployment Plan for Railway
1. Create Railway account at railway.app
2. Connect GitHub repo (mgorm34/GlossWerk)
3. Set environment variables: `GLOSSWERK_DEMO=true`, `ANTHROPIC_API_KEY=<key>`
4. Set start command: `cd scripts && streamlit run glosswerk_app.py --server.port $PORT --server.address 0.0.0.0`
5. Add custom domain: app.glosswerk.com
6. In Vercel DNS, add CNAME: `app` → Railway's provided domain
7. Estimated cost: ~$5/month

## 5. Demo-to-User Pipeline (Current Flow)

```
1. Prospect visits glosswerk.com
2. Fills out "Request Demo Access" form
3. Formspree captures submission → notifies you
4. You review request, generate invite code:
   python scripts/demo_auth.py generate --company "CompanyName" --email "their@email.com"
5. You email them the code from mjg@glosswerk.com
6. They go to app.glosswerk.com (NOT YET LIVE — needs Railway)
7. Enter invite code → auth gate validates → grants access
8. They upload a German patent .docx
9. Pipeline: term scan → glossary review → translate → QE triage → review → export
10. Usage tracked: 3 patents max, 14-day expiry, watermarked exports
```

**Gap at step 6**: Railway deployment needed. Temporary workaround: run Streamlit locally + ngrok tunnel.

## 6. Pending Work (Priority Order)

### Immediate (Before Outreach)
1. **Deploy Streamlit to Railway** at app.glosswerk.com — clients need to be able to actually use the demo
2. **Update Formspree notifications** to send to mjg@glosswerk.com
3. **Push latest code changes** (Formspree integration, patent limit=3, design updates, email fixes) — run `git add . && git commit && git push` from the glosswerk directory
4. **Test full pipeline end-to-end**: generate code → enter code → upload patent → translate → review → export

### Outreach
5. **Draft and send LSP outreach messages** (email, ProZ forum post, LinkedIn post)
6. **Generate invite codes** for each LSP you contact
7. **Target list**: ~15 LSPs that handle DE→EN patent translation (RWS, Welocalize, TransPerfect, SDL/RWS, etc.)

### Future Enhancements
8. **Document formatting function**: Let user describe font, spacing, etc. for final export doc
9. **Test on more patent domains**: mechanical engineering, chemistry, software, electrical
10. **Pricing implementation**: Move from free demo to paid pilot ($75-150/patent)
11. **Analytics dashboard**: Track demo usage, conversion rates, time-to-review metrics

## 7. Accounts & Credentials

| Service | Account | Notes |
|---------|---------|-------|
| GitHub | mgorm34 | Repo: mgorm34/GlossWerk, branch: main |
| Vercel | matthewjgorman34 | Project: gloss-werk, Hobby plan |
| Namecheap | — | Domain: glosswerk.com, expires Mar 2027, privacy on |
| Zoho Mail | mjg@glosswerk.com | Mail Lite plan |
| Formspree | matthewjgorman34@gmail.com | Form ID: xwvrgjye |
| Railway | NOT SET UP YET | — |

## 8. Key Design Decisions & Rationale

- **3 patent demo limit** (down from 5): Enough to evaluate the tool, scarce enough to create urgency for conversion
- **14-day expiry**: Prevents stale demo accounts, creates time pressure
- **Watermarked exports in demo**: Prevents using demo as a free production tool
- **Vercel + Railway split**: Vercel free tier for static landing page; Railway for dynamic Streamlit app. Total ~$5-15/month.
- **Invite codes (not self-serve)**: Lets you control who gets access, have a conversation before they try the tool, and track interest level
- **Formspree (not custom backend)**: Zero-maintenance form handling for a landing page. Free up to 50 submissions/month.
- **Zoho Mail (not forwarding)**: Separate professional inbox for business correspondence, keeps it distinct from personal Gmail

## 9. How to Resume Development

To pick up where this session left off:

```bash
# Navigate to project
cd glosswerk

# Check status
git status
git log --oneline -5

# Run the app locally
cd scripts
streamlit run glosswerk_app.py

# Generate a demo code
python demo_auth.py generate --company "TestCo" --email "test@test.com"

# List all codes
python demo_auth.py list
```

The most important next step is **Railway deployment**. Everything else (outreach, more testing, feature work) depends on having a live demo URL.
