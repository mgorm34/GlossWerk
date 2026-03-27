"""
Step 3b: Train T5-base on terminology-augmented patent correction data.
Uses the augmented database where inputs include terminology hints.

Input format: postedit: [MT output] || terms: Rastaufnahme=latching receptacle; Fig.=FIG.
Target: [Human reference translation]

Usage:
    python 03b_train_term_model.py --stage A
    python 03b_train_term_model.py --stage B
"""

import argparse
import glob
import os
import sqlite3
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)


class TermAugmentedAPEDataset(Dataset):
    """Dataset for terminology-augmented APE training."""

    def __init__(self, tokenizer, data, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        # The augmented MT already includes term hints:
        # "MT output || terms: X=Y; A=B"
        input_text = f"postedit: {row['mt_augmented']}"

        source = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        target = self.tokenizer(
            row["human_reference"], max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_data(db_path, split, mt_column="mt_opus_augmented"):
    """Load data from augmented SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    rows = cursor.execute(
        f"SELECT {mt_column} AS mt_augmented, ref AS human_reference, src, domain "
        f"FROM sentence_pairs WHERE split = ? AND {mt_column} IS NOT NULL",
        (split,),
    ).fetchall()

    data = [dict(r) for r in rows]
    conn.close()
    return data


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_db = os.path.join(project_root, "data", "glosswerk_patent_augmented.db")
    default_models = os.path.join(project_root, "models")

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["A", "B"],
                        help="A = train on opus-mt+terms, B = fine-tune on DeepL+terms")
    parser.add_argument("--db", type=str, default=default_db)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"GlossWerk Terminology-Augmented APE - Stage {args.stage}")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.stage == "A":
        output_dir = os.path.join(default_models, "patent_ape_term_stageA")
        model_name = "google-t5/t5-base"
        mt_column = "mt_opus_augmented"
        print(f"\nTraining on opus-mt + terminology \u2192 human reference")
        print(f"Base model: {model_name}")

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    elif args.stage == "B":
        stage_a_dir = os.path.join(default_models, "patent_ape_term_stageA", "final")
        output_dir = os.path.join(default_models, "patent_ape_term_stageB")
        mt_column = "mt_deepl_augmented"
        args.lr = 1e-4
        args.epochs = 5

        if not os.path.exists(stage_a_dir):
            print(f"ERROR: Stage A model not found at {stage_a_dir}")
            print("Run Stage A first.")
            sys.exit(1)

        print(f"\nFine-tuning on DeepL + terminology \u2192 human reference")
        print(f"Base model: Stage A ({stage_a_dir})")

        tokenizer = T5Tokenizer.from_pretrained(stage_a_dir)
        model = T5ForConditionalGeneration.from_pretrained(stage_a_dir)

    # Load data
    print(f"\nLoading data from: {args.db}")
    print(f"MT column: {mt_column}")
    train_data = load_data(args.db, "train", mt_column)
    val_data = load_data(args.db, "val", mt_column)

    if not train_data:
        print(f"ERROR: No training data found with {mt_column} column.")
        sys.exit(1)

    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    # Show a sample to verify format
    sample = train_data[0]
    print(f"\nSample input: postedit: {sample['mt_augmented'][:120]}")
    print(f"Sample target: {sample['human_reference'][:120]}")

    train_dataset = TermAugmentedAPEDataset(tokenizer, train_data)
    val_dataset = TermAugmentedAPEDataset(tokenizer, val_data)

    steps_per_epoch = len(train_data) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"\nSteps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=500 if args.stage == "A" else 100,
        bf16=True,
        eval_strategy="steps",
        eval_steps=500 if args.stage == "A" else 200,
        save_strategy="steps",
        save_steps=500 if args.stage == "A" else 200,
        save_total_limit=3,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if checkpoints:
        print(f"\nResuming from: {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=checkpoints[-1])
    else:
        print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}")
        trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to: {final_dir}")
    print(f"Training complete at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
