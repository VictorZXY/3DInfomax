from typing import Callable, List, Union

import torch
import torch.nn as nn


class BasicDecoder(nn.Module):

    def __init__(self, **kwargs):
        super(BasicDecoder, self).__init__()
        self.decoder = nn.Sigmoid()

    def forward(self, x):
        return self.decoder(torch.matmul(x, torch.transpose(x, 0, 1)))
