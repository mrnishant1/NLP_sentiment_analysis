import torch
import torch.nn as nn
import math


def sinusoidal_positional_encoding(seq_len, d_model, device=None):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe

class AttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [batch, seq, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn = torch.softmax(scores, dim=-1)      # normalize BEFORE weighting V
        return torch.matmul(attn, V)               # [batch, seq, d_model]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model) for _ in range(n_heads)])
        self.W_out = nn.Linear(d_model * n_heads, d_model, bias=False)

    def forward(self, x):
        head_outs = [h(x) for h in self.heads]
        concat = torch.cat(head_outs, dim=-1)
        return self.W_out(concat)


class EncoderBlock(nn.Module):
    """Standard transformer encoder block: MHA -> Add&Norm -> FFN -> Add&Norm.
        Encoder --> returns bidirection contextual aware embeddings"""
    def __init__(self, n_heads, d_model, ff_hidden, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.mha(x) #MHA
        x = self.norm1(x + self.dropout(attn_out)) #Add&Norm = x(embedding) + MHA with dropout
        ffn_out = self.ffn(x) #FFN
        x = self.norm2(x + self.dropout(ffn_out)) #Add&Norm = x + FFN with dropout
        return x
    
    
class MiniBERT(nn.Module): 
    def __init__(self, vocab_size, d_model, n_heads, seq_len, pad_idx, ff_hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx) 
        
        #Learnable -> Embedding weights 
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02) 
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0.0)

        self.register_buffer('pos_encoding', sinusoidal_positional_encoding(seq_len, d_model)) #register_buffer = register a buffer that should not be considered a model parameter
        self.encoder = EncoderBlock(n_heads, d_model, ff_hidden) #encoder --> returns bidirection contextual aware embeddings
        self.mlm_head = nn.Linear(d_model, vocab_size) #Neural layer for Sentiment fine-Tuning
        
    def encode(self, input_ids):
        """input_ids: [batch, seq] (long) -> [batch, seq, d_model]"""
        x = nn.Dropout(0.1)(self.embedding(input_ids)) + self.pos_encoding[: input_ids.shape[1]] #positional embeddings
        return self.encoder(x) 

    def forward(self, input_ids):
        """Returns MLM logits: [batch, seq, vocab_size]"""
        encoded = self.encode(input_ids) #bidirection contextual aware embeddings
        return self.mlm_head(encoded)