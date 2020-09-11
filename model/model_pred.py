import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
import math

class ModelPred(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.d = 2
        self.Q = nn.Parameter(self.init_uniform([self.d*2, config.attention_dim]))
        self.K = nn.ModuleList([nn.Linear(config.para_embedding_dim, config.attention_dim) for i in range(self.d*2)])
        self.h_fc = nn.Linear(config.para_embedding_dim*self.d*2, 1)
    
    def forward(self, query_inputs, option_inputs, types):
        batch_size = query_inputs.size(0)
        query_inputs = query_inputs.unsqueeze(1).repeat(1, 4, 1).view(batch_size*4, -1)
        h1, _ = self.encoder(query_inputs)
        h2, _ = self.encoder(option_inputs)
        h1_att1, h2_att1 = self.bi_attention_layer(h1, h2)
        h1_att2 = [self.attention_layer(self.Q[i], self.K[i](h1), h1_att1, query_inputs) for i in range(self.d)]
        h2_att2 = [self.attention_layer(self.Q[i+self.d], self.K[i+self.d](h2), h2_att1, option_inputs) for i in range(self.d)]
        h1_final, h2_final = torch.cat(h1_att2, dim=1), torch.cat(h2_att2, dim=1)
        h_final = torch.cat([h1_final, h2_final], dim=1)
        out = self.h_fc(h_final).view(-1, 4)
        return out