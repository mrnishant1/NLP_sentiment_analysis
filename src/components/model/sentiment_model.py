import torch.nn as nn
from .Bert import MiniBERT

class SentimentModel(nn.Module):
    def __init__(self, bert_model: MiniBERT, num_classes, dropout=0.2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.embedding.embedding_dim, num_classes)

    def forward(self, input_ids):
        encoded = self.bert.encode(input_ids)   # bidirectional context aware trained embeddings [batch, seq, d_model]
        cls = encoded[:, 0, :]                    # [batch, d_model]
        cls = self.dropout(cls)
        return self.classifier(cls)               # nnlayer output = [batch, num_classes]