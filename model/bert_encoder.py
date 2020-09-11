import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_encoder import BaseEncoder
from pytorch_pretrained_bert import BertModel, BertConfig

class BERTEncoder(BaseEncoder):        
    def __init__(self, config):
        super().__init__(config)
        self.encoder = config.encoder
        if config.train:
            bert = BertModel.from_pretrained('bert-base-chinese')
        else:
            bert = BertModel(BertConfig(21128))
        bert.cuda()
        if self.encoder == 'bert':
            self.bert = bert
        else:
            for p in bert.parameters():
                p.requires_grad = False
            BERTEncoder.bert = bert
    
    def encode_layer(self, input):
        if self.encoder == 'bert':
            h, h_pool = self.bert(input, attention_mask=(input>0), output_all_encoded_layers=False)
        else:
            with torch.no_grad():
                h, h_pool = BERTEncoder.bert(input, attention_mask=(input>0), output_all_encoded_layers=False)
                h, h_pool = h.detach(), h_pool.detach()
                h[input==0] = 0
                h_pool[input[:, 0]==0] = 0
        return h, h_pool
    
    def forward(self, input):
        h, h_pool = self.encode_layer(input)
        # h: [batch_size*para_num, seq_len, para_embedding_dim], h_pool: [batch_size*para_num, para_embedding_dim]
        return h, h_pool