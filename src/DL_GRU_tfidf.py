import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


cuda_idx = 5
data_dir = "data"
save_best = True
LR_Values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
# LR_Values = [5e-4]

# LR = 1e-4
weight_type = "Balanced"
EPOCHS = 50
results_dir_root = f"results/GRU_final"
best_result_dir = "results/best_model_test_results.csv"
dropout = 0.5
hidden_size = 64

train_df = pd.read_csv(os.path.join(data_dir,"train.csv"))
val_df   = pd.read_csv(os.path.join(data_dir,"val.csv"))
test_df  = pd.read_csv(os.path.join(data_dir,"test.csv"))
emotion_cols = ['joy','sadness','anger','fear','surprise','disgust','love','neutral']
X_train, y_train = train_df["clean_text"], train_df[emotion_cols]
X_val,   y_val   = val_df["clean_text"],   val_df[emotion_cols]
X_test,  y_test  = test_df["clean_text"],  test_df[emotion_cols]

print("Number of NaN values in training data:", X_train.isna().sum())
print("Number of NaN values in validation data:", X_val.isna().sum())
print("Number of NaN values in test data:", X_test.isna().sum())

mask_train = X_train.notna()
X_train = X_train[mask_train]
y_train = y_train[mask_train]

mask_val = X_val.notna()
X_val = X_val[mask_val]
y_val = y_val[mask_val]

mask_test = X_test.notna()
X_test = X_test[mask_test]
y_test = y_test[mask_test]

# Verify the NaN values are gone
print("\nAfter dropping NaN values:")
print("Number of NaN values in training data:", X_train.isna().sum())
print("Number of NaN values in validation data:", X_val.isna().sum())
print("Number of NaN values in test data:", X_test.isna().sum())

num_classes = len(emotion_cols)

# ------------------------------------------------
# 2️⃣ TF-IDF (1–3 gram, 30k features)
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
# 3️⃣ Dataset + Dataloader
# ------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]  # (1, features)

train_loader = DataLoader(EmotionDataset(X_train_tfidf, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(EmotionDataset(X_val_tfidf, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(EmotionDataset(X_test_tfidf, y_test), batch_size=64, shuffle=False)

# ------------------------------------------------
# 4️⃣ Model Definition
# ------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3, num_layers=1):
        super().__init__()
        self.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # can set True later for BiGRU
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, hn = self.GRU(x)  # hn: (num_layers, batch, hidden)
        out = hn[-1]                  # last layer hidden state
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# ------------------------------------------------
# 5️⃣ Setup Training
# ------------------------------------------------

device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

for LR in LR_Values :
    results_dir = os.path.join(results_dir_root, f"LR_{LR}")
    os.makedirs(results_dir, exist_ok=True) 

    
    model = GRUClassifier(input_size=X_train_tfidf.shape[1], hidden_size=hidden_size,dropout=dropout, num_classes=num_classes, num_layers=1).to(device)

    best_val = float('inf')
    patience, wait = 3, 0

    _state = {'best_val': best_val, 'wait': wait}

    def maybe_early_stop(val_loss):
        if val_loss < _state['best_val'] - 1e-4:  # small improvement
            _state['best_val'] = val_loss
            _state['wait'] = 0
            return False  # continue training
        else:
            _state['wait'] += 1
            return _state['wait'] >= patience  # stop if patience exhausted


        
    # pos_counts = y_train.sum(axis=0).values
    # neg_counts = len(y_train) - pos_counts
    # weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
    # weights = (neg_counts / pos_counts)
    # weights = weights / weights.mean()

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights).to(device))
    pos_counts = y_train.sum(axis=0).values.astype(np.float64)        # (#1s per class)
    total = len(y_train)
    neg_counts = total - pos_counts

    # avoid division by zero
    pos_counts = np.where(pos_counts == 0, 1.0, pos_counts)

    raw_w = neg_counts / pos_counts                   # imbalance ratio
    raw_w = np.clip(raw_w, 1.0, 10.0)                 # cap extremes (tame gradients)
    w = raw_w / raw_w.mean()                          # normalize around 1.0

    pos_weight = torch.tensor(w, dtype=torch.float32, device=device)

    if weight_type == "Balanced" :
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else :
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        
    # ------------------------------------------------
    # 6️⃣ Training Loop
    # ------------------------------------------------
    # def train_model(model, train_loader, val_loader, epochs=10):
    #     history = {'train_loss': [], 'val_loss': []}
    #     for epoch in range(epochs):
    #         model.train()
    #         total_loss = 0
    #         for X_batch, y_batch in train_loader:
    #             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #             optimizer.zero_grad()
    #             loss = criterion(model(X_batch), y_batch)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #         train_loss = total_loss / len(train_loader)

    #         # Validation
    #         model.eval()
    #         val_loss = 0
    #         with torch.no_grad():
    #             for X_batch, y_batch in val_loader:
    #                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #                 val_loss += criterion(model(X_batch), y_batch).item()
    #         val_loss /= len(val_loader)

    #         history['train_loss'].append(train_loss)
    #         history['val_loss'].append(val_loss)
    #         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    #     return history
    def train_model(model, train_loader, val_loader, epochs=EPOCHS):
        history = {'train_loss': [], 'val_loss': []}
        best_epoch = 0

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

            if val_loss < _state['best_val']:
                best_epoch = epoch + 1

            # early stopping
            if maybe_early_stop(val_loss):
                print(f"⏹️ Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                break

        print(f"✅ Best validation loss: {_state['best_val']:.4f} at epoch {best_epoch}")
        return history


    history = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # ------------------------------------------------
    # 7️⃣ Plot + Save Loss Graph
    # ------------------------------------------------
    

    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.title("GRU Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,"loss_plot.png"), dpi=300)
    plt.show()

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(results_dir,"training_history.csv"), index=False)

    # ------------------------------------------------
    # 8️⃣ Evaluation Function
    # ------------------------------------------------
    def evaluate_model_torch(model, loader, dataset_name="Test Set", threshold=0.5):
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > threshold).astype(int)
                y_pred.extend(preds)
                y_true.extend(y_batch.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Metrics
        subset_acc = accuracy_score(y_true, y_pred)  # exact match accuracy
        jaccard_acc = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
        micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"\n{dataset_name} @ threshold={threshold}")
        print(f"Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f} | Jaccard: {jaccard_acc:.4f}")

        return {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "subset_accuracy": subset_acc,
            "jaccard_accuracy": jaccard_acc
        }

    # ------------------------------------------------
    # 9️⃣ Evaluate and Save Results
    # ------------------------------------------------
    all_result = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        val_result  = evaluate_model_torch(model, val_loader, "Validation Set", threshold=t)
        test_result = evaluate_model_torch(model, test_loader, "Test Set", threshold=t)

        result = {"Model": "GRU (TF-IDF)", "Threshold" : t}
        result.update(test_result)
        all_result.append(result)

    result_df = pd.DataFrame(all_result)
    result_path = os.path.join(results_dir, "best_GRU_test_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"\nSaved results to {result_path} ")

    if save_best:
        best_score = result_df.loc[result_df["micro_f1"].idxmax()]  
        best_score = pd.DataFrame([best_score])
        
        if "Threshold" in best_score.index:
            best_score = best_score.drop("Threshold")

        cols = [
            "Model","micro_precision","micro_recall","micro_f1",
            "macro_precision","macro_recall","macro_f1",
            "subset_accuracy","jaccard_accuracy"
        ]
        best_score = best_score[cols]

        if os.path.exists(best_result_dir):
            all_best = pd.read_csv(best_result_dir)
        else:
            all_best = pd.DataFrame(columns=cols)


        all_best = pd.concat([all_best, best_score], ignore_index=True)


        all_best.to_csv(best_result_dir, index=False)
        print(f"✅ Appended best model result to {best_result_dir}")

    # ------------------------------------------------
    # 🔟 Confusion Matrices
    # ------------------------------------------------
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = (torch.sigmoid(model(X_batch)) > 0.5).int().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cms = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 4, figsize=(16,8))
    axes = axes.ravel()
    for idx, emotion in enumerate(emotion_cols):
        sns.heatmap(cms[idx], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(emotion)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,"confusion_matrices.png"), dpi=300)
    plt.show()
