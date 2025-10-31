# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['PathAggregation']

from torch import nn


class PathAggregation(nn.Module):
    def __init__(self, mode='avg'):
        super().__init__()
        self.mode = mode

    @staticmethod
    def average(path_embed):
        return path_embed.mean(1)

    def forward(self, path_embed):
        """
        Args:
            path_embed: torch.Tensor([n_node, k_path, d_path])

        Returns:
            path_embed: torch.Tensor([n_node, d_path])
        """
        if self.mode == 'avg':
            path_embed = self.average(path_embed)
        else:
            raise ValueError

        return path_embed
