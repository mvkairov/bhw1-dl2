import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset


def generate_square_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(x, pad_idx, device):
    if len(x.shape) == 2:
        idx_seq_len = x.shape[1]
    else:
        idx_seq_len = x.shape[0]
    idx_mask = generate_square_mask(idx_seq_len, device)
    idx_padding_mask = (x == pad_idx)
    return idx_mask, idx_padding_mask


class TinyStoriesDataset(Dataset):
    def __init__(self, data_file: str, sp_model_prefix: str = None,
                 vocab_size: int = 15000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 384):
        if not os.path.isfile(sp_model_prefix + '.model'):
            # train tokenizer if not trained yet
            print("Starting tokenizer train...")
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name, pad_id=5,
            )
            print("Finished tokenizer train")
        # load tokenizer from file

        print("Tokenizer?")
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model', )
        print("Tokenized.")

        print("Texts?")
        with open(data_file, encoding="utf-8") as file:
            texts = list(map(lambda x: x.strip(), file.readlines()))
        self.texts = texts
        print("Texted")

        print("Encoding?")
        self.indices = self.sp_model.encode(self.texts)
        print("Encoded.")

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()
        return self.sp_model.decode(ids)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        idx = [self.bos_id] + self.indices[item][:self.max_length - 2] + [self.eos_id]
        length = len(idx)
        idx += [self.pad_id] * (self.max_length - length)
        return idx, length
