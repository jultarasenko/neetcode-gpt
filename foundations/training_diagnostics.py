import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        with torch.no_grad():
            stats = []
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    dead = (x <= 0).all(dim=0)
                    stats.append({
                        'mean': round(torch.mean(x).item(), 4),
                        'std': round(torch.std(x).item(), 4),
                        'dead_fraction': round(dead.float().mean().item(), 4)
                    })
        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()

        stats = []
        for layer in model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                g = layer.weight.grad
                stats.append({
                    'mean': round(torch.mean(g).item(), 4),
                    'std': round(torch.std(g).item(), 4),
                    'norm': round(torch.norm(g).item(), 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(s['dead_fraction'] > 0.5 for s in activation_stats):
            return 'dead_neurons' 
        elif any(s['norm'] > 1000 or s['std'] > 10 for s in gradient_stats):
            return 'exploding_gradients' 
        elif gradient_stats[-1]['norm'] < 1e-5 or any(s['std'] < 0.1 for s in activation_stats):
            return 'vanishing_gradients' 
        else:
            return 'healthy'
        

