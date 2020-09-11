import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
import math

class ModelClassify(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.Q = nn.Parameter(self.init_uniform([2, config.attention_dim]))
        self.K = nn.ModuleList([nn.Linear(config.para_embedding_dim, config.attention_dim) for i in range(2)])
        self.h_fc = nn.Linear(config.para_embedding_dim*2, 1)
    
    def forward(self, query_inputs, option_inputs, types):
        batch_size = query_inputs.size(0)
        h1, _ = self.encoder(query_inputs)
        h2, _ = self.encoder(option_inputs)
        h1_att = self.attention_layer(self.Q[0], self.K[0](h1), h1, query_inputs)
        h2_att = self.attention_layer(self.Q[1], self.K[1](h2), h2, option_inputs)
        h2_max = torch.max(h2_att.view(batch_size, 4, -1), dim=1)[0]
        h_final = torch.cat([h1_att, h2_max], dim=1)
        out = self.h_fc(h_final)
        return out