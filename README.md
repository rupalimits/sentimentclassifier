# Sentiment Classification on Amazon Food Reviews

This project focuses on classifying nuanced sentiments in user-generated food reviews from Amazon using a Long Short-Term Memory (LSTM) neural network. The trained model is deployed as an interactive demo using **Gradio**, enabling real-time text sentiment inference.

---

## Objective

- Build a sentiment classification model that captures **contextual and emotional nuances** from food reviews.
- Train a deep learning model using the **Amazon Food Reviews dataset (~500,000 samples)**.
- Achieve high evaluation metrics, with a focus on **precision**.
- Deploy the trained model using an interactive Gradio web interface.

---

## Dataset

- **Dataset**: Amazon Fine Food Reviews
- **Size**: ~500,000 reviews
- **Features**:
  - `Text`: Raw review content
  - `Score`: Integer rating (1 to 5)
- **Labels**:
  - Mapped to 3 sentiment classes: `Negative`, `Neutral`, `Positive`

> Source: [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

---

## Preprocessing

- Removed stop words, punctuation, and special characters
- Applied tokenization and word embeddings (GloVe 100d)
- Converted review scores:
  - 1–2 → Negative
  - 3   → Neutral
  - 4–5 → Positive
- Split into training, validation, and test sets (80/10/10)

---

## Model Architecture

- **Model**: LSTM-based binary classifier
- **Embedding Layer**: Pretrained GloVe embeddings (100d)
- **Hidden Units**: 128
- **Dropout**: 0.3
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Epochs**: 10
- **Batch Size**: 64

---

## Evaluation Metrics

| Metric     | Score     |
|------------|-----------|
| Precision  | **88%**   |
| Recall     | 85%       |
| F1-Score   | 86.5%     |
| Accuracy   | 87%       |

---

## Technologies Used

- Python 3.9+
- PyTorch
- NLTK / SpaCy
- GloVe Embeddings
- Gradio for deployment
- Pandas & Matplotlib for EDA

---

## Deployment

The model is deployed using **Gradio**, allowing users to test real-time predictions through a web-based UI.

