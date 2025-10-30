import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------

cuda_idx = 7
data_dir = "data"
train_df = pd.read_csv(os.path.join(data_dir,"train.csv"))
val_df   = pd.read_csv(os.path.join(data_dir,"val.csv"))
test_df  = pd.read_csv(os.path.join(data_dir,"test.csv"))
results_dir = "results/rnn"
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
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# ------------------------------------------------
# 5️⃣ Setup Training
# ------------------------------------------------

device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = RNNClassifier(input_size=X_train_tfidf.shape[1], hidden_size=128, num_classes=num_classes).to(device)

pos_counts = y_train.sum(axis=0).values
neg_counts = len(y_train) - pos_counts
weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
weights = (neg_counts / pos_counts)
weights = weights / weights.mean()


print(weights)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# ------------------------------------------------
# 6️⃣ Training Loop
# ------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return history

history = train_model(model, train_loader, val_loader, epochs=20)

# ------------------------------------------------
# 7️⃣ Plot + Save Loss Graph
# ------------------------------------------------
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(history['train_loss'], label="Train Loss")
plt.plot(history['val_loss'], label="Val Loss")
plt.title("RNN Training vs Validation Loss")
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
            outputs = torch.sigmoid(model(X_batch)).cpu().numpy()
            preds = (outputs > threshold).astype(int)
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{dataset_name} Results:")
    print(f"Threshold : {threshold} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-micro: {f1_micro:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=emotion_cols, zero_division=0))

    return {
        "Threshold":threshold,
        "Accuracy": accuracy,
        "Precision_micro": precision,
        "Recall_micro": recall,
        "F1_micro": f1_micro,
        "F1_macro": f1_macro
    }

# ------------------------------------------------
# 9️⃣ Evaluate and Save Results
# ------------------------------------------------
all_result = []
for t in [0.2,0.3,0.4,0.5]:
    val_result  = evaluate_model_torch(model, val_loader, "Validation Set", threshold=t)
    test_result = evaluate_model_torch(model, test_loader, "Test Set", threshold=t)

    result = {"Model": "PyTorch RNN (TF-IDF, Multi-label)"}
    result.update(test_result)
    all_result.update(result)

result_df = pd.DataFrame([all_result])
result_df.to_csv(os.path.join(results_dir, "best_rnn_test_results.csv"), index=False)
print("\n✅ Saved results to ../results/rnn/best_rnn_test_results.csv")

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
