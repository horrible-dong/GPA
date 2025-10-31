# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['BehaviorEmbedding']

import torch
from torch import nn


class BehaviorEmbedding(nn.Module):
    def __init__(self, d_behav, l_path, d_path):
        super().__init__()
        self.proj = nn.Linear((d_behav * 2) * (l_path - 1), d_path, bias=False)

    def forward(self, g, paths):
        """
        Args:
            paths: torch.Tensor([n_node, k_path, l_path])

        Returns:
            behav_embed: torch.Tensor([n_node, k_path, d_path])
        """
        n_node, k_path, l_path = paths.shape
        behav_feat = torch.cat([g.ndata['behav'], torch.zeros_like(g.ndata['behav'][[-1]])])[paths]
        behav_feat = torch.cat([behav_feat[..., 0: l_path - 1, :], behav_feat[..., 1: l_path, :]], dim=-1)
        behav_embed = self.proj(behav_feat.flatten(-2))

        return behav_embed
