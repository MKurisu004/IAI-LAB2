import os, random, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from collections import Counter
from transformers import BertTokenizerFast

random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def read_corpus(path: str) -> List[Tuple[List[str], int]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            label, *tokens = line.strip().split()
            data.append((tokens, int(label)))
    return data

def build_vocab(corpus_files, min_freq: int = 1) -> Tuple[dict, dict]:
    cnt = Counter()
    for f in corpus_files:
        for tokens, _ in read_corpus(f):
            cnt.update(tokens)
    word2id = {PAD_TOKEN:0, UNK_TOKEN:1}
    for w,c in cnt.items():
        if c>= min_freq:
            word2id[w] = len(word2id)
        id2word = {i:w for w,i in word2id.items()}
    return word2id, id2word

class SentimentDataset(Dataset):
    def __init__(self, data, word2id: dict, max_len: int):
        self.max_len = max_len
        self.word2id = word2id
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        ids = [self.word2id.get(w, self.word2id[UNK_TOKEN]) for w in tokens]
        if len(ids) < self.max_len:
            ids += [self.word2id[PAD_TOKEN]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label,dtype=torch.long)

def load_word2vec(bin_path: str, word2id: dict, embed_dim: int) -> np.ndarray:
    kv = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    matrix = np.random.uniform(-0.1, 0.1, size=(len(word2id), embed_dim)).astype(np.float32)
    matrix[word2id[PAD_TOKEN]] = np.zeros(embed_dim)
    oov = 0
    for w, idx in word2id.items():
        if w in kv and kv[w].shape[0] == embed_dim:
            matrix[idx] = kv[w]
        else:
            oov += 1

    print(f"[emb] OOV words: {oov}/{len(word2id)}")
    return matrix

def make_loader(file, word2id, max_len, batch, shuffle):
    ds = SentimentDataset(read_corpus(file), word2id, max_len)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

def metric(tp, fp, fn, tn):
    precision = tp / (tp+fp+1e-8)
    recall    = tp / (tp+fn+1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (tp+tn) / (tp+fp+fn+tn+1e-8)
    return precision, recall, f1, acc


class BertDataset(Dataset):
    """
    将原始 token 序列重新 join 成句子 → 交给 HuggingFace Tokenizer。
    直接输出 (input_ids, attention_mask, label)
    """
    def __init__(self, samples, tokenizer: BertTokenizerFast, max_len: int):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        sentence = "".join(tokens)          # 中文直接拼即可；如需空格可改" ".join
        enc = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }
        return item

def make_loader_bert(file, tokenizer, max_len, batch, shuffle):
    ds = BertDataset(read_corpus(file), tokenizer, max_len)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)