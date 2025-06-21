import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')

# Load and preprocess text data
with open(r'C:\Users\dhaswin\Downloads\book (1)\sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

tokens = word_tokenize(text)
word_counts = Counter(tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# Prepare training data
sequence_length = 4
data = []

for i in range(len(tokens) - sequence_length):
    input_seq = tokens[i:i + sequence_length - 1]
    target = tokens[i + sequence_length - 1]
    if all(word in word2idx for word in input_seq + [target]):
        data.append((input_seq, target))

def encode(seq):
    return [word2idx[word] for word in seq]

encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx[target])) for inp, target in data]

# Define LSTM model
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(PredictiveKeyboard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of LSTM
        return out

# Training setup
model = PredictiveKeyboard(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 3  # Keep it small for demo

for epoch in range(epochs):
    total_loss = 0
    for x, y in encoded_data[:10000]:  # Limit samples for speed
        x = x.unsqueeze(0)  # batch size = 1
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model/predictive_model.pt")
print("✅ Model saved to model/predictive_model.pt")
