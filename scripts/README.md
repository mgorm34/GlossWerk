# Pipeline Scripts

The training and evaluation pipeline consists of six modular scripts, designed to be run sequentially.

## Usage

```bash
# 1. Download parallel corpora from OPUS
python 01_download_opus.py --domain patent --lang_pair de-en

# 2. Build training database with quality filtering
python 02_build_domain_db.py --domain patent --max_pairs 500000

# 3. Train APE model (Stage A)
python 03_train_patent_model.py --config ../configs/training_config.yaml

# 4. Generate DeepL translations for Stage B
python 04_deepl_corrections.py --budget 40000000  # character budget

# 5. Evaluate model performance
python 05_evaluate.py --model_path ../models/stage_a_patent --test_set patent

# 6. Log experiment results
python 06_track_progress.py --experiment stage_a_patent
```

## Script Details

| Script | Function | Runtime (RTX 5090) |
|---|---|---|
| `01_download_opus.py` | Downloads corpora via OPUS API. Supports EuroPat, EMEA, DGT, JRC-Acquis, KDE4, GNOME, ECB. | ~5-10 min |
| `02_build_domain_db.py` | Quality filtering, opus-mt baseline translation, TER computation, train/val/test split, SQLite storage. | ~1-2 hrs (94K pairs) |
| `03_train_patent_model.py` | T5 fine-tuning with bf16, gradient clipping, checkpoint resume. Supports Stage A and Stage B. | ~50 min (94K, 3 epochs) |
| `04_deepl_corrections.py` | DeepL API translation with character budget management, rate limiting, and progress tracking. | Depends on API tier |
| `05_evaluate.py` | BLEU/chrF scoring, per-domain breakdown, sample correction output, JSON export. | ~2-3 min |
| `06_track_progress.py` | Appends experiment metrics to CSV log for tracking progress across runs. | <1 sec |
