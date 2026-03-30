"""
HomeScout – PyTorch Fraud Classifier
Combines fine-tuned BERT embeddings + structured features → deep neural network
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3

STRUCTURED_FEATURES = [
    "price", "bedrooms", "bathrooms", "price_per_bed",
    "scam_keyword_count", "has_scam_keywords",
    "suspiciously_low_price", "suspiciously_high_price",
    "no_photos", "short_description", "photo_count",
    "fraud_score", "latitude", "longitude"
]

# ──────────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────────

class FraudDataset(Dataset):
    def __init__(self, structured, embeddings, labels):
        self.structured = torch.tensor(structured, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels     = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.structured[idx], self.embeddings[idx], self.labels[idx]


def load_data(csv_path="data/listings.csv"):
    print("📂 Loading data...")
    df         = pd.read_csv(csv_path)
    structured = df[STRUCTURED_FEATURES].fillna(0).values
    embeddings = np.load("models/bert_embeddings.npy")
    labels     = np.load("models/bert_labels.npy")

    scaler     = StandardScaler()
    structured = scaler.fit_transform(structured)
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"✅ Structured: {structured.shape} | Embeddings: {embeddings.shape}")
    print(f"   Fraud: {labels.sum():.0f} ({labels.mean()*100:.1f}%)")
    return structured, embeddings, labels


# ──────────────────────────────────────────────
# 2. NEURAL NETWORK
# ──────────────────────────────────────────────

class FraudClassifier(nn.Module):
    def __init__(self, structured_dim, embedding_dim, dropout=0.3):
        super().__init__()
        input_dim = structured_dim + embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)   # logits — sigmoid applied at eval
        )

    def forward(self, structured, embeddings):
        x = torch.cat([structured, embeddings], dim=1)
        return self.net(x).squeeze(1)


# ──────────────────────────────────────────────
# 3. HELPERS
# ──────────────────────────────────────────────

def recall_at_k(labels, probs, k=100):
    top_k_idx  = np.argsort(probs)[::-1][:k]
    top_k_true = labels[top_k_idx]
    return top_k_true.sum() / labels.sum() if labels.sum() > 0 else 0


def run_eval(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for s, e, y in loader:
            logits = model(s.to(DEVICE), e.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_prob  = np.array(all_probs)
    return y_true, y_pred, y_prob


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    structured, embeddings, labels = load_data()

    s_train, s_val, e_train, e_val, y_train, y_val = train_test_split(
        structured, embeddings, labels,
        test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(FraudDataset(s_train, e_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(FraudDataset(s_val,   e_val,   y_val),   batch_size=BATCH_SIZE)

    model = FraudClassifier(structured.shape[1], embeddings.shape[1]).to(DEVICE)
    print(f"\n🧠 Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Weighted loss for class imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(DEVICE)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    print(f"⚖️  Pos weight (fraud): {pos_weight.item():.2f}")
    print(f"\n🚀 Training for up to {EPOCHS} epochs...")

    best_f1, best_epoch, patience, patience_limit = 0, 0, 0, 5

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for s, e, y in train_loader:
            s, e, y = s.to(DEVICE), e.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(s, e), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Eval
        y_true, y_pred, y_prob = run_eval(model, val_loader)
        p  = precision_score(y_true, y_pred, zero_division=0)
        r  = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        scheduler.step(f1)

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, patience = f1, epoch, 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/fraud_classifier.pt")
            print(f"  💾 Best model saved! (F1: {best_f1:.4f})")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

    # Final report
    model.load_state_dict(torch.load("models/fraud_classifier.pt"))
    y_true, y_pred, y_prob = run_eval(model, val_loader)

    print(f"\n📊 Final Results (best epoch: {best_epoch})")
    print("=" * 55)
    print(f"  Precision  : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall     : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score   : {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC    : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Recall@100 : {recall_at_k(y_true, y_prob, k=100):.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    joblib.dump({
        "structured_dim": structured.shape[1],
        "embedding_dim":  embeddings.shape[1],
        "structured_features": STRUCTURED_FEATURES
    }, "models/classifier_config.pkl")

    print("\n🎉 Phase 2 complete! Run Phase 3 next (RAG Pipeline)")