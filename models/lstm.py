import torch
import torch.nn as nn
# LSTM sentiment classifier
class SentimentLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim=128,hidden_dim=128,num_layers=2,dropout=0.3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,batch_first=True,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_dim,1)
    def forward(self,x):
        embedded=self.embedding(x)
        _,(hidden,_)=self.lstm(embedded)
        output=self.dropout(hidden[-1])
        return self.fc(output).squeeze(1)