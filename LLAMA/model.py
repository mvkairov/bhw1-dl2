import torch
import torch.nn as nn

def generate_square_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(x, pad_idx, device):
    if len(x.shape) == 2:
        tgt_seq_len = x.shape[1]
    else:
        tgt_seq_len = x.shape[0]
    tgt_mask = generate_square_mask(tgt_seq_len, device)
    tgt_padding_mask = (x == pad_idx)
    return tgt_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.Tensor(10000.0)).item() / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LLAMA(nn.Module):
    def __init__(self, n_layers, d_model, nhead, vocab_size, dim_ff, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(embed_dim=d_model, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True), n_layers
        )
        self.lin = nn.Linear(d_model, vocab_size)
        print(f'{sum([torch.prod(torch.tensor(p.shape)) for p in self.parameters() if p.requires_grad])} params :)')
    
    def forward(self, input_ids, attention_mask, padding_mask):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer(x, attention_mask, padding_mask)
        return self.lin(x)
