from collections import Counter
# Build vocabulary from reviews
class Vocabulary:
    def __init__(self,min_freq=2):
        self.min_freq=min_freq
        self.pad_token="<PAD>"
        self.unk_token="<UNK>"
        self.word_to_index={self.pad_token:0,self.unk_token:1}
        self.index_to_word={0:self.pad_token,1:self.unk_token}
    def build(self,texts):
        counter=Counter()
        for text in texts:
            counter.update(text.split())
        for word,freq in counter.items():
            if freq>=self.min_freq:
                index=len(self.word_to_index)
                self.word_to_index[word]=index
                self.index_to_word[index]=word
    def encode(self,text):
        return [self.word_to_index.get(word,self.word_to_index[self.unk_token]) for word in text.split()]
    def __len__(self):
        return len(self.word_to_index)