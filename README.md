# Sentiment Analysis NLP

A Natural Language Processing project that classifies IMDb movie reviews as Positive or Negative using a CNN + Bidirectional LSTM model with pretrained GloVe embeddings.

## Dataset

- IMDb Movie Reviews Dataset
- 25,000 training reviews
- 25,000 testing reviews
- Binary classification: Positive / Negative

## Model

Review → Text Cleaning → Vocabulary → GloVe 100d → 1D CNN → BiLSTM → Dropout → Classifier

## Technologies

- Python
- PyTorch
- Hugging Face Datasets
- GloVe
- NLP
- Deep Learning

## Results

| Model | Test Accuracy |
|---|---:|
| Basic LSTM | 71.10% |
| BiLSTM | 83.10% |
| BiLSTM + GloVe | 84.83% |
| CNN + BiLSTM + GloVe | **84.99%** |

## Prediction

The model can classify custom movie reviews and provide a confidence score.

Example:

This movie was absolutely amazing and I loved every minute of it.

Prediction: Positive
Confidence: 96.17%

## Project Structure

Sentiment-Analysis-NLP/
├── models/
│   └── lstm.py
├── dataset.py
├── vocabulary.py
├── train.py
├── test.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore

## How to Run

Install dependencies:

pip install -r requirements.txt

Train:

python train.py

Test:

python test.py

Predict:

python predict.py

