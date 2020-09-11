import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_encoder import BaseEncoder

class BiLSTMEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.word_embedding = nn.Embedding(config.vocab_num, config.word_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(config.word_embedding_dim, config.para_embedding_dim//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
    
    def encode_layer(self, input, x):
        h, _ = self.lstm(x)
        h = self.dropout(h)
        h[input==0] = 0
        h_t = h.transpose(-2, -1)
        h_pool = F.max_pool1d(h_t, h_t.size(-1)).squeeze(-1)
        return h, h_pool
    
    def forward(self, input):
        x = self.word_embedding(input)
        h, h_pool = self.encode_layer(input, x)
        # h: [batch_size*para_num, seq_len, para_embedding_dim], h_pool: [batch_size*para_num, para_embedding_dim]
        return h, h_pool