"""
LoRA Fine-tuning for Domain-Specific APE

Trains a LoRA adapter on top of the existing T5 APE model.
Automatically uses mt_deepl if available, falls back to mt_opus.

Usage:
  python 15_train_lora.py --db C:\\glosswerk\\data\\medical_training.db --ipc A61 --output C:\\glosswerk\\models\\a61_medical_lora
"""

import argparse
import sqlite3
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType


class APEDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mt, ref = self.data[idx]
        input_text = f"postedit: {mt}"

        input_enc = self.tokenizer(
            input_text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        target_enc = self.tokenizer(
            ref, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        labels = target_enc.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels": labels,
        }


def load_data(db_path, ipc_prefix="A61"):
    conn = sqlite3.connect(db_path)

    # Check which MT column has data
    deepl_count = conn.execute(
        "SELECT COUNT(*) FROM domain_pairs WHERE mt_deepl IS NOT NULL AND mt_deepl != ''"
    ).fetchone()[0]
    opus_count = conn.execute(
        "SELECT COUNT(*) FROM domain_pairs WHERE mt_opus IS NOT NULL AND mt_opus != ''"
    ).fetchone()[0]

    if deepl_count > 0:
        mt_col = "mt_deepl"
        print(f"  Using mt_deepl ({deepl_count} rows)")
    elif opus_count > 0:
        mt_col = "mt_opus"
        print(f"  Using mt_opus ({opus_count} rows)")
    else:
        print("  ERROR: No MT data found!")
        return [], []

    train_rows = conn.execute(
        f"SELECT {mt_col}, ref FROM domain_pairs WHERE {mt_col} IS NOT NULL AND {mt_col} != '' AND ipc_code LIKE ? AND split='train'",
        (f"{ipc_prefix}%",)
    ).fetchall()

    val_rows = conn.execute(
        f"SELECT {mt_col}, ref FROM domain_pairs WHERE {mt_col} IS NOT NULL AND {mt_col} != '' AND ipc_code LIKE ? AND split='val'",
        (f"{ipc_prefix}%",)
    ).fetchall()

    conn.close()

    if len(val_rows) < 100 and len(train_rows) > 200:
        val_rows = train_rows[:200]
        train_rows = train_rows[200:]

    return train_rows, val_rows


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=r"C:\glosswerk\models\patent_ape_stageA\final")
    parser.add_argument("--db", default=r"C:\glosswerk\data\domain_patent_training.db")
    parser.add_argument("--output", default=r"C:\glosswerk\models\a61f_lora")
    parser.add_argument("--ipc", default="A61F")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\n--- Loading {args.ipc} data from {args.db} ---")
    train_data, val_data = load_data(args.db, args.ipc)
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")

    if len(train_data) == 0:
        print("ERROR: No training data found!")
        return

    print(f"\n--- Loading base model ---")
    tokenizer = T5TokenizerFast.from_pretrained(args.base_model)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    print(f"\n--- Configuring LoRA (rank={args.rank}, alpha={args.lora_alpha}) ---")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    train_dataset = APEDataset(train_data, tokenizer)
    val_dataset = APEDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    print(f"\n--- Training ---")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {args.lr}")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            if (i + 1) % 100 == 0:
                avg = epoch_loss / n_batches
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {i+1}/{len(train_loader)} | Loss: {avg:.4f} | Time: {elapsed:.0f}s")

        avg_train_loss = epoch_loss / n_batches
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time

        print(f"  Epoch {epoch+1} done | Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f} | Time: {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  ** Saved best model (val_loss={val_loss:.4f}) **")

    print(f"\n--- Done ---")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {args.output}")


if __name__ == "__main__":
    main()