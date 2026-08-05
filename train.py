import os
import json
import numpy as np
import torch

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# -----------------------------
# Load all intent files
# -----------------------------
INTENTS_FOLDER = "intents"

all_intents = []

for file_name in os.listdir(INTENTS_FOLDER):
    if file_name.endswith(".json"):
        file_path = os.path.join(INTENTS_FOLDER, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            if "intents" in data:
                all_intents.extend(data["intents"])

print(f"Loaded {len(all_intents)} intents.")

# -----------------------------
# Prepare training data
# -----------------------------
all_words = []
tags = []
xy = []

ignore_words = ["?", "!", ".", ","]

for intent in all_intents:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"Vocabulary Size : {len(all_words)}")
print(f"Tags            : {len(tags)}")

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


# ==============================
# Next Part
# ==============================

from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 8
hidden_size = 16
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 500

print("===================================")
print("Training Personal AI")
print("Input Size :", input_size)
print("Hidden Size:", hidden_size)
print("Output Size:", output_size)
print("Epochs     :", num_epochs)
print("===================================")

# -----------------------------
# Dataset
# -----------------------------
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()

train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)


# ==============================
# Next Part
# ==============================

import torch.nn as nn

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Training
# -----------------------------
for epoch in range(num_epochs):
    for (words, labels) in train_loader:

        words = words.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.4f}")

print("\nTraining completed!")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, "data/data.pth")

print("Model saved to data/data.pth")
