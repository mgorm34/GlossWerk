# Technical Notes

Implementation notes and lessons learned during GlossWerk development.

## Training Stability: bf16 vs fp16 on Blackwell

The RTX 5090 (Blackwell, sm_120) natively supports both fp16 and bf16 mixed precision. However, T5-base training with fp16 causes **silent loss divergence** — loss appears to decrease normally for the first few hundred steps, then jumps to NaN without warning. The resulting model outputs only commas regardless of input.

Root cause: fp16's limited exponent range (5 bits) causes overflow in T5's relative position bias computation. bf16's 8-bit exponent handles the same values without overflow. This is architecture-specific — other models may not exhibit this behavior.

**Fix**: Always use bf16 on Blackwell for T5 family models. Set `bf16=True` in HuggingFace TrainingArguments and verify with a short training run before committing to full epochs.

Loss curve with bf16: 7.08 → 3.1 over 3 epochs (steady decrease, no divergence).

## Data Quality: The EUbookshop Failure

The first training attempt used OPUS EUbookshop corpus, which appeared to have adequate volume for DE-EN training. The resulting model produced BLEU 0.00 — complete garbage output.

Investigation revealed systematic misalignment in the EUbookshop parallel data. Many "aligned" sentence pairs were not translations of each other but adjacent sentences from the same document. TER between source translations and references was extremely high, confirming the misalignment.

**Lesson**: Always validate corpus quality before training. TER-based filtering (threshold ≤ 0.5) between MT output and human references catches misalignment effectively. Domain-specific corpora (EuroPat for patents) are more reliably aligned than general-purpose collections.

## CUDA Toolkit Compatibility

The RTX 5090 Laptop GPU uses Blackwell architecture (compute capability sm_120). As of early 2026:

- CUDA 12.8 (cu128) via PyTorch nightly: full support, correct kernel selection
- Earlier CUDA versions: may compile but produce warnings and fallback to slower code paths
- Install: `pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128`

Verify after install:
```python
import torch
print(torch.cuda.get_device_capability())  # Should show (12, 0)
print(torch.version.cuda)                   # Should show 12.8
```

## opus-mt Throughput

Helsinki-NLP/opus-mt-de-en achieves ~31 sentences/second on RTX 5090 with default batch inference settings. For the full 94K filtered dataset, baseline translation completes in ~50 minutes. Scaling to 500K+ pairs should take ~4.5 hours.

## TER Filtering Rationale

Translation Error Rate (TER) measures the edit distance between MT output and human reference, normalized by reference length. We use TER ≤ 0.5 as the inclusion threshold:

- **TER < 0.1**: MT output nearly identical to reference — little for the model to learn
- **TER 0.1-0.5**: Meaningful differences the model can learn to correct
- **TER > 0.5**: Likely misalignment, or the MT output is so bad that corrections become unreliable signal

This filtering reduced the initial 200K candidate pairs to 94K usable training examples. The aggressive cutoff is intentional — cleaner data consistently outperforms larger noisy data for APE tasks.

## SQLite for Training Data

Training data is stored in SQLite rather than flat files for practical reasons:

- Atomic train/val/test splits without file duplication
- Easy querying for analysis (e.g., distribution of TER scores, sentence lengths)
- Portable — single file for the entire dataset
- Fast random access during inspection and debugging

The schema stores source (DE), MT output (opus-mt or DeepL), human reference (EN), TER score, and split assignment for each pair.
