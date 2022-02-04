
from typing import Callable, List, Union

import torch
import torch.nn as nn

from models.base_layers import MLP


class BasicCritic(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim,
                 layers,
                 dropout=0,
                 **kwargs):
        super(BasicCritic, self).__init__()
        self.criticise = MLP(in_dim=in_dim, hidden_size=hidden_dim,
                             mid_batch_norm=True, out_dim=out_dim,
                             dropout=dropout,
                             layers=layers)

    def forward(self, x):
        return self.criticise(x)