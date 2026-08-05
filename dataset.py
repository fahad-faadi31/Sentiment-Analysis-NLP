import re
import torch
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from vocabulary import Vocabulary
# Clean review text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"<br\s*/?>"," ",text)
    text=re.sub(r"[^a-z\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text
# IMDb dataset class
class IMDBDataset(Dataset):
    def __init__(self,data,vocab,max_length=200):
        self.data=data
        self.vocab=vocab
        self.max_length=max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        text=clean_text(self.data[index]["text"])
        label=self.data[index]["label"]
        tokens=self.vocab.encode(text)
        tokens=tokens[:self.max_length]
        tokens=tokens+[self.vocab.word_to_index[self.vocab.pad_token]]*(self.max_length-len(tokens))
        return torch.tensor(tokens,dtype=torch.long),torch.tensor(label,dtype=torch.float)
# Load IMDb dataset
def load_imdb_dataset():
    dataset=load_dataset("stanfordnlp/imdb")
    dataset=dataset.map(lambda x:{"text":clean_text(x["text"])})
    train_validation=dataset["train"].train_test_split(test_size=0.2,seed=42)
    dataset["train"]=train_validation["train"]
    dataset["validation"]=train_validation["test"]
    return dataset
# Create vocabulary
def build_vocab(dataset):
    vocab=Vocabulary(min_freq=2)
    vocab.build(dataset["train"]["text"])
    return vocab
if __name__=="__main__":
    dataset=load_imdb_dataset()
    vocab=build_vocab(dataset)
    train_dataset=IMDBDataset(dataset["train"],vocab)
    validation_dataset=IMDBDataset(dataset["validation"],vocab)
    test_dataset=IMDBDataset(dataset["test"],vocab)
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    validation_loader=DataLoader(validation_dataset,batch_size=64)
    test_loader=DataLoader(test_dataset,batch_size=64)
    texts,labels=next(iter(train_loader))
    print("Vocabulary Size:",len(vocab))
    print("Train Size:",len(train_dataset))
    print("Validation Size:",len(validation_dataset))
    print("Test Size:",len(test_dataset))
    print("Input Shape:",texts.shape)
    print("Label Shape:",labels.shape)