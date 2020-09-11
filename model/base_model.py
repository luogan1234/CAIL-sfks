import torch
import torch.nn as nn
import torch.nn.functional as F
from model.textcnn_encoder import TextCNNEncoder
from model.bilstm_encoder import BiLSTMEncoder
from model.bert_encoder import BERTEncoder
import math

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.encoder == 'textcnn':
            self.encoder = TextCNNEncoder(config)
        if config.encoder == 'bilstm':
            self.encoder = BiLSTMEncoder(config)
        if config.encoder in ['bert', 'bert_freeze']:
            self.encoder = BERTEncoder(config)
    
    def init_uniform(self, shape):
        if isinstance(shape, int):
            shape = [shape]
        stdv = 1. / math.sqrt(shape[-1])
        w = torch.empty(shape)
        nn.init.uniform_(w, -stdv, stdv)
        return w
    
    def average_layer(self, h, x):
        h_mean = []
        for i in range(h.size(0)):
            r = max(torch.sum(x[i]>0).item(), 1)
            m = torch.mean(h[i, :r], 0)
            h_mean.append(m)
        h_mean = torch.stack(h_mean)
        return h_mean
    
    def bi_attention_layer(self, V1, V2):
        alphas = torch.tanh(torch.bmm(V1, V2.transpose(1, 2)))
        O1 = torch.bmm(F.softmax(alphas, 1), V2)
        O2 = torch.bmm(F.softmax(alphas, 0).transpose(1, 2), V1)
        return O1, O2
    
    def attention_layer(self, Q, K, V, x):
        alphas = torch.matmul(torch.tanh(K), Q)
        mask = x>0
        mask[~torch.any(mask, 1)] = True
        alphas[~mask] = float('-inf')
        alphas = F.softmax(alphas, 1)
        outs = torch.bmm(alphas.unsqueeze(1), V).squeeze(1)
        return outs
   
    def forward(self, query_inputs, option_inputs, types):
        raise NotImplementedError