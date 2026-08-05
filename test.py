import torch
from torch.utils.data import DataLoader
from dataset import load_imdb_dataset,build_vocab,IMDBDataset
from models.lstm import SentimentLSTM
# Evaluation configuration
BATCH_SIZE=64
MAX_LENGTH=200
EMBEDDING_DIM=100
HIDDEN_DIM=128
DROPOUT=0.3
# Select device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data
dataset=load_imdb_dataset()
vocab=build_vocab(dataset)
test_dataset=IMDBDataset(dataset["test"],vocab,MAX_LENGTH)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE)
# Load model
model=SentimentLSTM(len(vocab),EMBEDDING_DIM,HIDDEN_DIM,DROPOUT).to(device)
model.load_state_dict(torch.load("sentiment_lstm.pth",map_location=device))
model.eval()
# Evaluate model
correct=0
total=0
with torch.no_grad():
    for texts,labels in test_loader:
        texts=texts.to(device)
        labels=labels.to(device)
        outputs=model(texts)
        predictions=(torch.sigmoid(outputs)>=0.5).float()
        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)
accuracy=correct/total
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")