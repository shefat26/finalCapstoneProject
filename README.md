# 🎥 Movie Genre Classifier

This project uses machine learning to classify movie genres based on their plot summaries.

---

## 📌 Overview

- **Input:** Movie plot summaries  
- **Output:** Predicted genre  
- **Goal:** Automate genre tagging using NLP and ML

---

## 📁 Dataset

- ~15,000 movie entries  
- Key columns:  
  - `Title`  
  - `Plot_summary`  
  - `Genres`  

Data was cleaned, normalized, and prepared for modeling.

---

## 🧪 Models Used

- **Logistic Regression** – Accuracy: ~49.6%  
- **Random Forest** – Accuracy: ~45.6%  
- **Custom LSTM (Deep Learning)** – Trained using Keras

---

## 🔧 Features & Preprocessing

- Cleaned and lowercased text  
- Removed noise (HTML tags, special characters, etc.)  
- Tokenization + sentence and word counts  
- TF-IDF vectorization (top 1000–5000 words)  
- Extracted top keywords per plot  
- Genre-level keyword profiling

---

## 📊 Visualizations

- Genre frequency plots  
- Genre co-occurrence heatmap  
- Word clouds and top N-grams  
- Confusion matrices for model evaluation

---

