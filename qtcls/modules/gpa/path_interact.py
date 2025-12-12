# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['PathInteraction']

import copy

from torch import nn


class PathInteraction(nn.Module):
    def __init__(self, d_path, n_head=1, n_attn=2, d_ffn=1024, dropout=0.1, activation=nn.ReLU, pre_norm=False):
        super().__init__()
        interaction_layer = InteractionLayer(d_path, n_head, d_ffn, dropout, activation)
        norm_pre = nn.LayerNorm(d_path) if pre_norm else nn.Identity()
        self.interaction = Interaction(interaction_layer, n_attn, norm_pre)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, path_embed, behav_embed=None):
        """
        Args:
            path_embed: torch.Tensor([n_node, k_path, d_path])
            behav_embed: torch.Tensor([n_node, k_path, d_path]) or None

        Returns:
            path_embed: torch.Tensor([n_node, k_path, d_path])
        """
        path_embed = self.interaction(path_embed, behav_embed)

        return path_embed


class Interaction(nn.Module):
    def __init__(self, interaction_layer, n_attn=2, norm_pre=None):
        super().__init__()
        self.interaction_layers = nn.ModuleList([copy.deepcopy(interaction_layer) for _ in range(n_attn)])
        self.norm_pre = norm_pre

    def forward(self, path_embed, behav_embed=None):
        """
        Args:
            path_embed: torch.Tensor([n_node, k_path, d_path])
            behav_embed: torch.Tensor([n_node, k_path, d_path]) or None

        Returns:
            path_embed: torch.Tensor([n_node, k_path, d_path])
        """
        path_embed = path_embed.transpose(0, 1)

        if behav_embed is not None:
            behav_embed = behav_embed.transpose(0, 1)

        path_embed = self.norm_pre(path_embed)

        for interaction_layer in self.interaction_layers:
            path_embed = interaction_layer(path_embed, behav_embed)

        return path_embed.transpose(0, 1)


class InteractionLayer(nn.Module):
    def __init__(self, d_path, n_head=1, d_ffn=1024, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_path, n_head, dropout=dropout)

        self.linear1 = nn.Linear(d_path, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_path)

        self.norm1 = nn.LayerNorm(d_path)
        self.norm2 = nn.LayerNorm(d_path)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, path_embed, behav_embed=None):
        """
        Args:
            path_embed: torch.Tensor([n_node, k_path, d_path])
            behav_embed: torch.Tensor([n_node, k_path, d_path]) or None

        Returns:
            path_embed: torch.Tensor([n_node, k_path, d_path])
        """
        query = key = path_embed if behav_embed is None else path_embed + behav_embed
        path_embed2 = self.attention(query, key, value=path_embed)[0]
        path_embed = path_embed + self.dropout1(path_embed2)
        path_embed = self.norm1(path_embed)
        path_embed2 = self.linear2(self.dropout(self.activation(self.linear1(path_embed))))
        path_embed = path_embed + self.dropout2(path_embed2)
        path_embed = self.norm2(path_embed)

        return path_embed
