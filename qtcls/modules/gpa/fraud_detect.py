# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['FraudDetection']

from torch import nn


class FraudDetection(nn.Module):
    def __init__(self, d_path, num_classes):
        super().__init__()
        self.predictor = nn.Linear(d_path, num_classes)

    def forward(self, path_embed):
        """
        Args:
            path_embed: torch.Tensor([n_node, d_path])

        Returns:
            logits: torch.Tensor([n_node, num_classes])
        """
        logits = self.predictor(path_embed)

        return logits
