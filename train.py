import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import load_imdb_dataset,build_vocab,IMDBDataset
from torch.utils.data import DataLoader
from models.lstm import SentimentLSTM
# Training configuration
BATCH_SIZE=64
EPOCHS=10
LEARNING_RATE=0.0005
MAX_LENGTH=200
EMBEDDING_DIM=128
HIDDEN_DIM=128
DROPOUT=0.3
PATIENCE=2
# Select device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data
dataset=load_imdb_dataset()
vocab=build_vocab(dataset)
train_dataset=IMDBDataset(dataset["train"],vocab,MAX_LENGTH)
validation_dataset=IMDBDataset(dataset["validation"],vocab,MAX_LENGTH)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
validation_loader=DataLoader(validation_dataset,batch_size=BATCH_SIZE)
# Initialize model
model=SentimentLSTM(len(vocab),EMBEDDING_DIM,HIDDEN_DIM,DROPOUT).to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=Adam(model.parameters(),lr=LEARNING_RATE)
best_accuracy=0
patience_counter=0
for epoch in range(EPOCHS):
    model.train()
    total_loss=0
    correct=0
    total=0
    for texts,labels in train_loader:
        texts=texts.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(texts)
        loss=criterion(outputs,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()
        predictions=(torch.sigmoid(outputs)>=0.5).float()
        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)
    train_accuracy=correct/total
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for texts,labels in validation_loader:
            texts=texts.to(device)
            labels=labels.to(device)
            outputs=model(texts)
            predictions=(torch.sigmoid(outputs)>=0.5).float()
            correct+=(predictions==labels).sum().item()
            total+=labels.size(0)
    validation_accuracy=correct/total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} Train Accuracy: {train_accuracy:.4f} Validation Accuracy: {validation_accuracy:.4f}")
    if validation_accuracy>best_accuracy:
        best_accuracy=validation_accuracy
        patience_counter=0
        torch.save(model.state_dict(),"sentiment_lstm.pth")
        print("Best model saved!")
    else:
        patience_counter+=1
        if patience_counter>=PATIENCE:
            print("Early stopping!")
            break
print(f"Best Validation Accuracy: {best_accuracy:.4f}")