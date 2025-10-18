import re
import pickle
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.freq_threshold = freq_threshold
        self.word_count = {}
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        
        self.word_count = dict(frequencies)
    
    def tokenize(self, text):
        # Convert to lowercase
        text = text.lower()

        # Step 1: Protect numbers with decimals and percentages
        # Replace "26.3%" with "26.3_percent" (treat as a single token)
        text = re.sub(r'(\d+\.\d+)%', r'\1_percent', text)  # 26.3% → 26.3_percent
        text = re.sub(r'(\d+)%', r'\1_percent', text)       # 5% → 5_percent
        text = re.sub(r'(\d+)\.(\d+)', r'\1.\2', text)      # 1.5 → 1.5 (unchanged)

        # Step 2: Remove unwanted punctuation (except protected cases)
        # Keep: letters, numbers, underscores, and protected tokens (e.g., 26.3_percent)
        text = re.sub(r'[^\w\s.]', '', text)  # Allow dots (.) in numbers

        # Step 3: Split into tokens
        tokens = text.split()

        return tokens
    
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    
    def save_vocab(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'itos': self.itos,
                'stoi': self.stoi,
                'word_count': self.word_count
            }, f)
    
    @classmethod
    def load_vocab(cls, path):
        vocab = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            vocab.itos = data['itos']
            vocab.stoi = data['stoi']
            vocab.word_count = data['word_count']
        return vocab