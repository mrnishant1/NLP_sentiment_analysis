import torch 
import torch.nn as nn
from ..preprocessing import mask_input_ids
from config import DEVICE
device = DEVICE
import numpy as np


def train_mlm(model, input_ids_all, epochs, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    n = len(input_ids_all)
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            raw_batch = input_ids_all[batch_idx]

            masked_batch, labels_batch = [], []
            for row in raw_batch:
                m_ids, m_labels = mask_input_ids(row)
                masked_batch.append(m_ids)
                labels_batch.append(m_labels)

            x = torch.tensor(masked_batch, dtype=torch.long, device=device)
            y = torch.tensor(labels_batch, dtype=torch.long, device=device)

            logits = model(x)                       # [batch, seq, vocab]
            loss = loss_fn(logits.permute(0, 2, 1), y) # Permute = Bcz crossENtropy need [batch, vocab, seq]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}  MLM Loss: {avg_loss:.4f}")

    return loss_history

def train_sentiment(model, X_train, Y_train, epochs, batch_size, lr, class_weights):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    n = len(X_train)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n)
        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            x = torch.tensor(X_train[batch_idx], dtype=torch.long, device=device)
            y = torch.tensor(Y_train[batch_idx], dtype=torch.long, device=device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}  Sentiment Loss: {avg_loss:.4f}")

    return loss_history
