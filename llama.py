import torch
from torch import Tensor
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LLAMA(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size, ff_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.classification = nn.Linear(embed_dim, vocab_size)
    
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer(x, attention_mask, padding_mask)
        return self.classification(x)