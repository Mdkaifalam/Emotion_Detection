# 🧠 Emotion Detection from Text

## 🎯 Objective
The goal of this project is to build a text-based emotion detection system that classifies text into six primary emotions:

**joy**, **sadness**, **anger**, **fear**, **surprise**, and **disgust**.

The project compares two approaches:

1. **Classical Machine Learning** — TF-IDF + Logistic Regression / SVM  
2. **Deep Learning** — Word Embeddings + BiLSTM / GRU  

This comparison provides insights into trade-offs between interpretability, performance, and computational efficiency.

---

## ⚙️ Project Pipeline

### 1. Data Collection  
- **Dataset:** GoEmotions (Google, via HuggingFace)  
- Reddit comments labeled with 27 fine-grained emotions  
- For simplicity, six core emotions are selected for this project

### 2. Data Preprocessing  
- Convert text to lowercase  
- Remove URLs, mentions, punctuation, and stopwords  
- Tokenize sentences  
- Split into train, validation, and test sets

### 3. Feature Extraction  
- **Classical ML:** TF-IDF Vectorization  
- **Deep Learning:** Word2Vec / GloVe embeddings or trainable embeddings using PyTorch  

### 4. Model Architectures

#### 🔹 Classical ML
- Logistic Regression and Support Vector Machine (SVM)  
- Fast and interpretable models for baseline performance  

#### 🔹 Deep Learning
- BiLSTM / GRU-based model architecture  
- Embedding → Recurrent Layer → Dense (Softmax) Output  
- Optimizer: Adam  
- Loss: Cross-Entropy  

### 5. Evaluation Metrics  
- Accuracy  
- Macro F1-Score  
- Precision & Recall  
- Confusion Matrix Visualization  

### 6. Explainability & Visualization  
- **SHAP / LIME:** For understanding classical model predictions  
- **Attention Visualization:** For deep learning model interpretability  
- Comparative visualization between classical vs deep models  

---

## 🧰 Tech Stack

| Component | Tools |
|------------|--------|
| **Data** | HuggingFace Datasets (GoEmotions) |
| **ML / DL Frameworks** | scikit-learn, PyTorch |
| **Text Processing** | NLTK, Regex, TF-IDF |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP, LIME |
| **Environment** | Python 3.10+, GPU (CUDA) |

---

## 📂 Project Structure

```bash
Emotion_Detection/
│
├── data/                    # Raw or processed datasets
├── src/
│   ├── preprocess.py         # Text cleaning and tokenization
│   ├── classical_model.py    # TF-IDF + LogisticRegression/SVM
│   ├── deep_model.py         # BiLSTM / GRU model
│   ├── train_classical.py    # Training loop for classical ML model
│   ├── train_deep.py         # Training loop for deep model
│   ├── evaluate.py           # Metrics, plots, confusion matrix
│   ├── explain.py            # SHAP/LIME interpretation
│   └── config.py             # Constants and hyperparameters
│
├── results/                  # Stores evaluation outputs
├── notebooks/                # Data exploration / analysis
├── app/demo.py               # Inference demo script
├── requirements.txt
├── setup.sh
└── README.md
