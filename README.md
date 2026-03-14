# GlossWerk

**Domain-specific neural post-editing for machine translation.**

GlossWerk is an Automatic Post-Editing (APE) engine that corrects domain-specific errors in machine translation output. General-purpose MT engines like DeepL produce fluent translations but consistently mishandle specialized terminology, compound nouns, and formatting conventions in technical domains. GlossWerk sits as a correction layer on top of these engines, making targeted improvements without disrupting overall translation quality.

Current focus: German → English patent translation (EuroPat v3 corpus).

## Results

Stage A evaluation on 47,332 held-out patent test sentences:

| System | BLEU | chrF |
|---|---|---|
| opus-mt (baseline) | 47.64 | 69.12 |
| opus-mt + GlossWerk | 54.35 (+6.71) | 73.02 (+3.90) |

DeepL cross-engine evaluation (2,450 sentences, from earlier smaller-scale run):

| System | BLEU | chrF |
|---|---|---|
| DeepL (baseline) | 53.78 | 73.74 |
| DeepL + GlossWerk | 55.61 (+1.83) | 74.45 (+0.71) |

The model improves DeepL output **despite never seeing DeepL translations during training**. Correction patterns learned from opus-mt errors transfer across MT engines.

### Example Corrections

| Source (DE) | opus-mt Output | GlossWerk Output | Reference |
|---|---|---|---|
| ...die gewünschten Quadratursignale erzeugen lassen. | ...the desired **square signals** can be generated. | ...the desired **quadrature signals** can be generated. | ...the desired quadrature signals can be obtained. |
| Die Rastaufnahme 31 besteht aus einer Auflagewand 35... | The **rest picture** 31 consists of a support wall 35... | The **latching recess** 31 consists of a support wall 35... | The latching receptacle 31 includes a bearing wall 35... |
| Die Düsen können gemäß Fig. | The nozzles can be according to **Fig.** | The nozzles according to **FIG.** | The nozzles can be arranged according to FIG. |

The model actively edits 90% of input sentences, targeting terminology corrections, patent formatting conventions, and domain-specific compound nouns.

## Architecture

GlossWerk uses a two-stage fine-tuning approach on T5-base (220M parameters):

```
                    ┌─────────────────────────────────────────────┐
                    │            GlossWerk Pipeline                │
                    │                                             │
  German Source ──► │  MT Engine ──► APE Model ──► Corrected EN   │
                    │  (DeepL)      (T5-base)                     │
                    └─────────────────────────────────────────────┘
```

**Stage A** (complete): Fine-tune T5-base on opus-mt translations → human references. The model learns general post-editing behavior for patent text.

**Stage B** (planned): Further fine-tune on DeepL translations → human references. This adapts the model to DeepL's specific error distribution for production use.

### Data Pipeline

Raw parallel data from OPUS EuroPat v3 (19.7M DE-EN sentence pairs) is processed through multi-stage quality filtering:

1. **Length & character filtering** — 10-500 chars, >40% alphabetic, length ratio 0.3-3.0x
2. **Baseline translation** — All German sources translated via opus-mt (~31 sent/sec on RTX 5090)
3. **TER filtering** — Translation Error Rate between opus-mt output and human reference; pairs with TER > 0.5 discarded
4. **Train/val/test split** — 94,472 clean pairs split 80/10/10, stored in SQLite

## Project Structure

```
glosswerk/
├── scripts/
│   ├── 01_download_opus.py      # Multi-domain corpus downloader via OPUS API
│   ├── 02_build_domain_db.py    # Quality filtering, translation, TER computation, SQLite storage
│   ├── 03_train_patent_model.py # Two-stage T5 training with bf16, gradient clipping, checkpointing
│   ├── 04_deepl_corrections.py  # DeepL API integration with character budget management
│   ├── 05_evaluate.py           # BLEU/chrF scoring, per-domain breakdown, sample corrections
│   └── 06_track_progress.py     # CSV-based experiment logging
├── configs/
│   └── training_config.yaml     # Hyperparameters and training configuration
├── results/
│   └── stage_a_patent.json      # Evaluation metrics and sample outputs
├── docs/
│   └── technical_notes.md       # Implementation notes and lessons learned
├── requirements.txt
└── README.md
```

## Infrastructure

| Component | Specification |
|---|---|
| Hardware | NVIDIA RTX 5090 (24GB VRAM, Blackwell sm_120) |
| Framework | PyTorch 2.12 nightly (CUDA 12.8 / cu128), HuggingFace Transformers |
| Base Model | google-t5/t5-base (220M parameters) |
| Training | bf16 mixed precision, batch_size=16, grad_accum=4, lr=3e-4, 3 epochs |
| Data | OPUS EuroPat v3 — 19.7M DE-EN pairs, ~47K test after quality filtering |
| Training Time | ~50 minutes on RTX 5090 |

## Technical Notes

A few things learned during development that might be useful to others:

- **bf16 vs fp16 on Blackwell**: fp16 mixed precision causes silent NaN loss divergence with T5-base on RTX 5090. bf16 trains stably due to its larger exponent range. Loss decreased steadily from 7.08 → 3.1.
- **Data quality matters more than quantity**: Initial training on OPUS EUbookshop corpus produced BLEU 0.00 due to systematic source-reference misalignment. Switching to domain-specific EuroPat with TER-based filtering resolved this entirely.
- **CUDA/Blackwell compatibility**: RTX 5090 (sm_120) requires PyTorch nightly with CUDA 12.8 (cu128). Earlier toolkit versions produce compatibility warnings and suboptimal performance.

## Roadmap

- [ ] Scale patent training data from 94K → 500K+ pairs
- [ ] Stage B: Fine-tune on DeepL output distributions
- [ ] Medical domain model (OPUS EMEA, 1.1M pairs)
- [ ] EN → DE reverse direction
- [ ] Legal domain (DGT, JRC-Acquis) and technical domains (KDE4, GNOME)
- [ ] T5-large (770M) evaluation
- [ ] Human evaluation with professional patent translators

## License

MIT

## Contact

Matt Gorman — [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [Email](mailto:your@email.com)
