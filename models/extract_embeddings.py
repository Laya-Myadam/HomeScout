"""
HomeScout – Vanilla BERT Embedding Extractor
Extracts [CLS] embeddings from pretrained DistilBERT (no fine-tuning)
Used to establish baseline before LoRA fine-tuning
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Using device: {DEVICE}")


# ──────────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────────

class ListingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
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
            "attention_mask": encoding["attention_mask"].squeeze()
        }


# ──────────────────────────────────────────────
# 2. LOAD & PREPARE DATA
# ──────────────────────────────────────────────

def load_data(path="data/listings.csv"):
    print("📂 Loading listings.csv...")
    df = pd.read_csv(path)
    df["description"] = df["description"].fillna("")

    # Enrich text with structured context for richer embeddings
    df["text"] = df.apply(lambda r: (
        f"Price: ${r['price']}/month. "
        f"Bedrooms: {r['bedrooms']}. "
        f"Bathrooms: {r['bathrooms']}. "
        f"Description: {str(r['description'])[:200]}"
    ), axis=1)

    print(f"✅ Loaded {len(df)} listings")
    return df


# ──────────────────────────────────────────────
# 3. EXTRACT [CLS] EMBEDDINGS
# ──────────────────────────────────────────────

def extract_embeddings(texts, model, tokenizer):
    dataset = ListingDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # [CLS] token = first token of last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1) * BATCH_SIZE}/{len(texts)} listings...")

    return np.vstack(all_embeddings)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    df     = load_data("data/listings.csv")
    texts  = df["text"].tolist()
    labels = df["is_fraud"].values

    # Load vanilla BERT — no fine-tuning
    print(f"\n🤖 Loading vanilla {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"✅ Model loaded")

    # Extract embeddings
    print(f"\n📦 Extracting embeddings for {len(texts)} listings...")
    print("   (this takes ~3-5 mins on CPU)")
    embeddings = extract_embeddings(texts, model, tokenizer)

    print(f"\n✅ Embeddings shape: {embeddings.shape}")  # (49352, 768)

    # Save
    os.makedirs("models", exist_ok=True)
    np.save("models/bert_embeddings.npy", embeddings)
    np.save("models/bert_labels.npy",     labels)
    print("💾 Saved:")
    print("   models/bert_embeddings.npy")
    print("   models/bert_labels.npy")

    print("\n🎉 Done! Run train_model.py next for baseline results.")