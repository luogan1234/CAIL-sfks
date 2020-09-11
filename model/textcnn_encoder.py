import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_encoder import BaseEncoder

class TextCNNEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.word_embedding = nn.Embedding(config.vocab_num, config.word_embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(1, config.num_filters, (k, config.token_embedding_dim), padding=(k//2, 0)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(0.2)
    
    def encode_layer(self, input, x):
        x = x.unsqueeze(1)
        out = []
        for conv in self.convs:
            o = F.relu(conv(x)).squeeze(-1)
            out.append(o)
        h = torch.cat(out, 1)  # [batch_size*para_num, para_embedding_dim, seq_len]
        h = self.dropout(h.transpose(-2, -1))
        h[input==0] = 0
        h_t = h.transpose(-2, -1)
        h_pool = F.max_pool1d(h_t, h_t.size(-1)).squeeze(-1)
        return h, h_pool
    
    def forward(self, input):
        x = self.word_embedding(input)
        h, h_pool = self.encode_layer(input, x)
        # h: [batch_size*para_num, seq_len, para_embedding_dim], h_pool: [batch_size*para_num, para_embedding_dim]
        return h, h_pool