import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        result = torch.randn(fan_out, fan_in) * std
        return torch.round(result, decimals = 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        result = torch.randn(fan_out, fan_in) * std
        return torch.round(result, decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        dims = [input_dim] + [hidden_dim] * num_layers
        layers = [nn.Linear(d_in, d_out, bias=False)
                for d_in, d_out in zip(dims[:-1], dims[1:])]

        torch.manual_seed(0)
        for linear in layers:
            if init_type == 'xavier':
                nn.init.xavier_normal_(linear.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(linear.weight)
            else:
                nn.init.normal_(linear.weight)

        x = torch.randn(1, input_dim)  
        stds = []
        for linear in layers:
            x = torch.relu(linear(x))
            stds.append(round(x.std().item(), 2))
        return stds