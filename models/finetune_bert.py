"""
HomeScout – BERT Fine-tuning with LoRA/PEFT
Fine-tunes DistilBERT on listing descriptions for fraud classification
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 5
LR           = 2e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Using device: {DEVICE}")

# ──────────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────────

class ListingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(path="data/listings.csv"):
    print("📂 Loading data...")
    df = pd.read_csv(path)
    df["description"] = df["description"].fillna("")

    # Enrich text with structured context for better fine-tuning
    df["text"] = df.apply(lambda r: (
        f"Price: ${r['price']}/month. "
        f"Bedrooms: {r['bedrooms']}. "
        f"Bathrooms: {r['bathrooms']}. "
        f"Description: {str(r['description'])[:200]}"
    ), axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["is_fraud"].tolist(),
        test_size=0.2,
        stratify=df["is_fraud"],
        random_state=42
    )
    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"   Fraud in train: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    return X_train, X_val, y_train, y_val


# ──────────────────────────────────────────────
# 2. MODEL WITH LoRA
# ──────────────────────────────────────────────

def build_lora_model():
    print(f"\n🤖 Loading {MODEL_NAME} with LoRA/PEFT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                          # rank — lower = fewer params
        lora_alpha=16,                # scaling factor
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────
# 3. CLASS WEIGHTS (handle imbalance)
# ──────────────────────────────────────────────

def compute_class_weights(y_train):
    fraud_count = sum(y_train)
    legit_count = len(y_train) - fraud_count
    total       = len(y_train)
    w_legit     = total / (2 * legit_count)
    w_fraud     = total / (2 * fraud_count)
    weights     = torch.tensor([w_legit, w_fraud], dtype=torch.float).to(DEVICE)
    print(f"⚖️  Class weights — Legit: {w_legit:.2f} | Fraud: {w_fraud:.2f}")
    return weights


# ──────────────────────────────────────────────
# 4. TRAIN
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
            loss       = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            probs  = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds  = outputs.logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    auc       = roc_auc_score(all_labels, all_probs)

    return total_loss / len(loader), precision, recall, f1, auc


# ──────────────────────────────────────────────
# 5. EXTRACT EMBEDDINGS (for PyTorch classifier)
# ──────────────────────────────────────────────

def extract_embeddings(model, loader):
    """Extract [CLS] embeddings from fine-tuned BERT for downstream classifier."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Get hidden states from base model
            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_emb.cpu().numpy())

    return np.vstack(embeddings)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    X_train, X_val, y_train, y_val = load_data("data/listings.csv")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets & loaders
    train_dataset = ListingDataset(X_train, y_train, tokenizer)
    val_dataset   = ListingDataset(X_val,   y_val,   tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    # Model
    model = build_lora_model().to(DEVICE)

    # Loss with class weights
    class_weights = compute_class_weights(y_train)
    loss_fn       = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\n🚀 Fine-tuning for {EPOCHS} epochs...")
    best_f1, best_epoch = 0, 0
    patience, patience_limit = 0, 2  # early stopping

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_loss, precision, recall, f1, auc = eval_epoch(model, val_loader, loss_fn)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Precision:  {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1    = f1
            best_epoch = epoch
            patience   = 0
            os.makedirs("models", exist_ok=True)
            model.save_pretrained("models/bert_lora")
            tokenizer.save_pretrained("models/bert_lora")
            print(f"  💾 Best model saved (F1: {best_f1:.4f})")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Best F1: {best_f1:.4f} at epoch {best_epoch}")

    # Save embeddings for PyTorch classifier
    print("\n📦 Extracting embeddings for downstream classifier...")
    full_dataset   = ListingDataset(X_train + X_val, y_train + y_val, tokenizer)
    full_loader    = DataLoader(full_dataset, batch_size=BATCH_SIZE)
    all_embeddings = extract_embeddings(model, full_loader)
    all_labels     = np.array(y_train + y_val)

    np.save("models/bert_embeddings.npy", all_embeddings)
    np.save("models/bert_labels.npy",     all_labels)
    print("💾 Embeddings saved to models/bert_embeddings.npy")
    print("\n🎉 LoRA fine-tuning complete! Run train_model.py next.")