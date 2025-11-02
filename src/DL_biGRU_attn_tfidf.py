import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    multilabel_confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, jaccard_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ------------------------------------------------
# Config
# ------------------------------------------------
cuda_idx = 5
data_dir = "data"
save_best = False
# LR_Values = [5e-4]
LR_Values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
weight_type = "Balanced"
EPOCHS = 50
results_dir_root = f"results/BiGRU_Attention"
best_result_dir = "results/best_model_test_results.csv"
dropout = 0.5
hidden_size = 64
bidir = True

# ------------------------------------------------
# Load Data
# ------------------------------------------------
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
val_df   = pd.read_csv(os.path.join(data_dir, "val.csv"))
test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))
emotion_cols = ['joy','sadness','anger','fear','surprise','disgust','love','neutral']

X_train, y_train = train_df["clean_text"], train_df[emotion_cols]
X_val,   y_val   = val_df["clean_text"],   val_df[emotion_cols]
X_test,  y_test  = test_df["clean_text"],  test_df[emotion_cols]

# Drop NaN rows
X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
X_val, y_val     = X_val.dropna(),   y_val.loc[X_val.dropna().index]
X_test, y_test   = X_test.dropna(),  y_test.loc[X_test.dropna().index]

num_classes = len(emotion_cols)

# ------------------------------------------------
# TF-IDF (1–3 gram, 30k features)
# ------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    stop_words='english'
)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_val_tfidf   = tfidf.transform(X_val).toarray()
X_test_tfidf  = tfidf.transform(X_test).toarray()

# ------------------------------------------------
# Dataset + Dataloader
# ------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx].unsqueeze(0), self.y[idx]

train_loader = DataLoader(EmotionDataset(X_train_tfidf, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(EmotionDataset(X_val_tfidf, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(EmotionDataset(X_test_tfidf, y_test), batch_size=64, shuffle=False)

# ------------------------------------------------
# 🧠 BiGRU + Attention Model
# ------------------------------------------------
class BiGRU_AttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3, num_layers=1, bidir=True):
        super().__init__()
        self.bidir = bidir
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir
        )

        # Attention layer
        self.attn = nn.Linear(hidden_size * (2 if bidir else 1), 1)
        self.fc1 = nn.Linear(hidden_size * (2 if bidir else 1), 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  # out: [batch, seq_len, hidden*2]
        attn_weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # [batch, seq_len]
        attn_output = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)  # weighted sum [batch, hidden*2]
        out = torch.relu(self.fc1(attn_output))
        out = self.dropout(out)
        return self.fc2(out)

# ------------------------------------------------
# Training Setup
# ------------------------------------------------
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

for LR in LR_Values:
    results_dir = os.path.join(results_dir_root, f"LR_{LR}")
    os.makedirs(results_dir, exist_ok=True)

    model = BiGRU_AttentionClassifier(
        input_size=X_train_tfidf.shape[1],
        hidden_size=hidden_size,
        dropout=dropout,
        num_classes=num_classes,
        num_layers=1,
        bidir=bidir
    ).to(device)

    best_val = float('inf')
    patience, wait = 3, 0
    _state = {'best_val': best_val, 'wait': wait}

    def maybe_early_stop(val_loss):
        if val_loss < _state['best_val'] - 1e-4:
            _state['best_val'] = val_loss
            _state['wait'] = 0
            return False
        else:
            _state['wait'] += 1
            return _state['wait'] >= patience

    pos_counts = y_train.sum(axis=0).values.astype(np.float64)
    total = len(y_train)
    neg_counts = total - pos_counts
    pos_counts = np.where(pos_counts == 0, 1.0, pos_counts)
    raw_w = np.clip(neg_counts / pos_counts, 1.0, 10.0)
    w = raw_w / raw_w.mean()
    pos_weight = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # ------------------------------------------------
    # Training
    # ------------------------------------------------
    def train_model(model, train_loader, val_loader, epochs=EPOCHS):
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(X_batch), y_batch).item()
            val_loss /= len(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if maybe_early_stop(val_loss):
                print("⏹️ Early stopping triggered.")
                break

        return history

    history = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # ------------------------------------------------
    # Plot Loss
    # ------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.title("BiGRU + Attention Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_plot.png"), dpi=300)
    plt.show()

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------
    def evaluate_model(model, loader, threshold=0.5):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                probs = torch.sigmoid(model(X_batch)).cpu().numpy()
                preds = (probs > threshold).astype(int)
                y_pred.extend(preds)
                y_true.extend(y_batch.numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return {
            "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "subset_accuracy": accuracy_score(y_true, y_pred),
            "jaccard_accuracy": jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        }

    all_result = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        test_result = evaluate_model(model, test_loader, threshold=t)
        result = {"Model": "BiGRU + Attention (TF-IDF)", "Threshold": t}
        result.update(test_result)
        all_result.append(result)

    result_df = pd.DataFrame(all_result)
    result_path = os.path.join(results_dir, "best_BiGRU_Attention_test_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"\nSaved results to {result_path}")

print("🎯 BiGRU + Attention training & evaluation done!")
