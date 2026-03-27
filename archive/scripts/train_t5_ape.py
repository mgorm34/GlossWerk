"""
GlossWerk APE - Phase 2: T5-base Fine-tuning Script
Trains a T5-base model to post-edit machine translation output.

Input:  SQLite database with columns: source_de, mt_output, human_reference, domain, split
Output: Fine-tuned T5-base model saved to checkpoint directory

Usage:
    conda activate glosswerk
    python train_t5_ape.py --db "C:\glosswerk\data\glosswerk_ape (Newest).db" --output_dir "C:\glosswerk\models\t5_ape_model_v3"
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)


class APEDataset(Dataset):
    """Dataset for Automatic Post-Editing training."""

    def __init__(self, tokenizer, data, max_source_length=256, max_target_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        mt_output = row["mt_output"]
        human_reference = row["human_reference"]

        # Input format: "postedit: [MT output]"
        input_text = f"postedit: {mt_output}"

        # Tokenize input
        source = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target = self.tokenizer(
            human_reference,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored in loss
        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_data_from_db(db_path):
    """Load training data from SQLite database."""
    print(f"Loading data from: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check what tables/columns exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables found: {tables}")

    # Try to find the right table and columns
    # Adjust these based on your actual schema
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"  {table}: {columns}")

    # Load data - adjust query based on your actual schema
    # Common patterns from the roadmap:
    #   source_de, mt_output, human_reference, domain, split
    try:
        train_data = [
            dict(row)
            for row in cursor.execute(
                "SELECT mt_opus AS mt_output, ref AS human_reference, src, domain FROM sentence_pairs WHERE split = 'train' AND ter <= 0.5"
            ).fetchall()
        ]
        val_data = [
            dict(row)
            for row in cursor.execute(
                "SELECT mt_opus AS mt_output, ref AS human_reference, src, domain FROM sentence_pairs WHERE split = 'val' AND ter <= 0.5"
            ).fetchall()
        ]
        test_data = [
            dict(row)
            for row in cursor.execute(
                "SELECT mt_opus AS mt_output, ref AS human_reference, src, domain FROM sentence_pairs WHERE split = 'test'"
            ).fetchall()
        ]
    except Exception as e:
        print(f"\nCould not load with default query: {e}")
        print("\nPlease update the SQL queries in this script to match your database schema.")
        print("The script expects columns: mt_output, human_reference, and a split column.")
        print("\nShowing first few rows from each table for reference:")
        for table in tables:
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            rows = cursor.fetchall()
            if rows:
                print(f"\n  {table} sample:")
                for row in rows:
                    print(f"    {dict(row)}")
        conn.close()
        sys.exit(1)

    conn.close()
    print(f"Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Train T5-base APE model")
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help='Path to SQLite database (e.g., "C:\\glosswerk\\data\\glosswerk_ape (Newest).db")',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:\\glosswerk\\models\\t5_ape_model_v3",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-t5/t5-base",
        help="Base model to fine-tune (default: t5-base)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max sequence length for tokenizer"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and model but don't train (test setup)",
    )
    args = parser.parse_args()

    # Check GPU
    print("=" * 60)
    print("GlossWerk APE - T5 Training Script")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected! Training will be very slow on CPU.")
    print()

    # Load data
    train_data, val_data, test_data = load_data_from_db(args.db)

    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = T5ForConditionalGeneration.from_pretrained(args.resume_from)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.0f}M")

    # Create datasets
    train_dataset = APEDataset(tokenizer, train_data, args.max_length, args.max_length)
    val_dataset = APEDataset(tokenizer, val_data, args.max_length, args.max_length)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: ~{len(train_dataset) // args.batch_size}")
    print(f"Total steps: ~{len(train_dataset) // args.batch_size * args.epochs}")

    if args.dry_run:
        print("\n--- DRY RUN: Setup looks good! Remove --dry_run to train. ---")

        # Test one batch
        sample = train_dataset[0]
        print(f"\nSample input shape: {sample['input_ids'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")

        # Decode to verify
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        label_ids = sample["labels"].clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        target_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        print(f"\nSample input:  {input_text[:100]}...")
        print(f"Sample target: {target_text[:100]}...")
        return

    # Training arguments - optimized for RTX 5090 (32GB VRAM)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=f"glosswerk_ape_{timestamp}",
        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        # fp16 for faster training on RTX 5090
        fp16=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=1000,
        # Saving
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=False,
        #metric_for_best_model="eval_loss",
        #greater_is_better=False,
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        report_to="none",
        # Performance
        dataloader_num_workers=0,
        gradient_accumulation_steps=4,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[],
    )

    # Train
    print(f"\n{'=' * 60}")
    print(f"Starting training at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}\n")

    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to: {final_dir}")

    # Evaluate on test set if available
    if test_data:
        test_dataset = APEDataset(tokenizer, test_data, args.max_length, args.max_length)
        results = trainer.evaluate(test_dataset)
        print(f"\nTest results: {results}")

    print(f"\nTraining complete at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
