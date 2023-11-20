import torch
from torch import Tensor
import torch.nn as nn
import copy

nn.TransformerDecoderLayer
nn.TransformerDecoder

# class DecoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, ff_dim, dropout, batch_first=True, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
#         self.attn_dropout = nn.Dropout(dropout)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, ff_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_dim, embed_dim),
#             nn.Dropout(dropout)
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#     def forward(self, x, attention_mask, padding_mask):
#         x, _ = self.attention(x, x, x, attn_mask=attention_mask, key_padding_mask=padding_mask)
#         x = self.norm1(x + self.attn_dropout(x))
#         x = self.norm2(x + self.feed_forward(x))
#         return x

# class Decoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    
#     def forward(self, x, attention_mask, padding_mask):
#         for layer in self.decoder:
#             x = layer(x, attention_mask, padding_mask)
#         return x

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
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LLAMA(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size, ff_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.classification = nn.Linear(embed_dim, vocab_size)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer(x, attention_mask, padding_mask)
        return self.classification(x)