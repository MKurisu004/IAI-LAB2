import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, ker_sizes, pretrained_emb=None, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.convs = nn.ModuleList([    
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2) for k in ker_sizes
        ])
        self.fc = nn.Linear(num_filters * len(ker_sizes), 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   
        emb = self.embedding(x).transpose(1, 2)  
        features = [F.relu(conv(emb)).max(dim=-1)[0] for conv in self.convs]
        out = torch.cat(features, dim=1)
        return self.fc(self.dropout(out))
    
class RNN_Base(nn.Module):
    def __init__(self, cell, vocab_size, embed_dim, hidden, layers, dropout, pretrained=None, bidirectional = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.rnn = cell(input_size = embed_dim,
                        hidden_size = hidden,
                        num_layers = layers,
                        batch_first = True,
                        dropout = dropout if layers >1 else 0,
                        bidirectional = bidirectional)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ =self.rnn(emb)
        feat = out[:,-1,:]
        return self.fc(self.dropout(feat))
    
class RNN_LSTM(RNN_Base):
    def __init__(self, *args, **kw):
        super().__init__(nn.LSTM, *args, **kw)

class RNN_GRU(RNN_Base):
    def __init__(self, *args, **kw):
        super().__init__(nn.GRU, *args, **kw)

class MLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, max_len, pretrained=None, dropout=0.3):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.classifier = nn.Sequential(
            nn.Linear(max_len * embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        emb = self.embedding(x)                              
        flat = emb.view(emb.size(0), -1)                      
        return self.classifier(flat) 
        
class BERTClassifier(nn.Module):
    """
    取 BERT [CLS] 向量 → Dropout → Linear → 2 类 softmax
    """
    def __init__(self, bert_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

    def forward(self, batch):
        # batch 是一个 dict，包括 input_ids / attention_mask
        out = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        cls = out.last_hidden_state[:, 0]          # [B, H]
        return self.classifier(cls)                # [B, 2]

