import torch
from dataset import clean_text,load_imdb_dataset,build_vocab
from models.lstm import SentimentLSTM
# Model configuration
MAX_LENGTH=200
EMBEDDING_DIM=100
HIDDEN_DIM=128
DROPOUT=0.3
# Select device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dataset and vocabulary
dataset=load_imdb_dataset()
vocab=build_vocab(dataset)
# Load model
model=SentimentLSTM(len(vocab),EMBEDDING_DIM,HIDDEN_DIM,DROPOUT).to(device)
model.load_state_dict(torch.load("sentiment_lstm.pth",map_location=device))
model.eval()
# Predict sentiment
def predict_sentiment(text):
    text=clean_text(text)
    tokens=vocab.encode(text)
    tokens=tokens[:MAX_LENGTH]
    tokens=tokens+[vocab.word_to_index[vocab.pad_token]]*(MAX_LENGTH-len(tokens))
    inputs=torch.tensor(tokens,dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output=model(inputs)
        probability=torch.sigmoid(output).item()
    sentiment="Positive" if probability>=0.5 else "Negative"
    confidence=probability if sentiment=="Positive" else 1-probability
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence*100:.2f}%")
# User input
text=input("Enter a review: ")
predict_sentiment(text)