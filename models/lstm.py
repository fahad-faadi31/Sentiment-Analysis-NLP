import torch
import torch.nn as nn
# Bidirectional LSTM sentiment classifier
class SentimentLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim=100,hidden_dim=128,dropout=0.3,embedding_weights=None):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_dim*2,1)
    def forward(self,x):
        embedded=self.embedding(x)
        _,(hidden,_)=self.lstm(embedded)
        output=torch.cat((hidden[-2],hidden[-1]),dim=1)
        output=self.dropout(output)
        return self.fc(output).squeeze(1)