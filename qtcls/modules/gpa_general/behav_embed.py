# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['BehaviorEmbedding']

import torch
from torch import nn


class BehaviorEmbedding(nn.Module):
    def __init__(self, d_behav, l_path, d_path):
        super().__init__()
        self.scaler = nn.Parameter(torch.tensor(1.))
        self.proj = nn.Linear(d_behav * l_path, d_path, bias=False)

    def forward(self, g, paths):
        """
        Args:
            paths: torch.Tensor([n_node, k_path, l_path])

        Returns:
            behav_embed: torch.Tensor([n_node, k_path, d_path])
        """
        behav_feat = torch.cat([g.ndata['behav'], torch.zeros_like(g.ndata['behav'][[-1]])])[paths]
        behav_embed = self.proj(behav_feat.flatten(-2) * self.scaler)

        return behav_embed
