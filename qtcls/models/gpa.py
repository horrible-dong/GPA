# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['GPA', 'gpa']

from torch import nn

from ..modules.behav_embed import BehaviorEmbedding
from ..modules.path_aggregate import PathAggregation
from ..modules.path_embed import PathEmbedding
from ..modules.path_interact import PathInteraction
from ..modules.path_sample import PathSampling


class GPA(nn.Module):
    def __init__(self,
                 d_node, d_behav, n_path, k_path, l_path, d_path,
                 n_head=1, n_encoder=2, d_feedforward=1024, dropout=0.1, activation='relu',
                 aggr_mode='avg',
                 num_classes=2):
        super().__init__()
        self.path_sampling = PathSampling(n_path, k_path, l_path)
        self.path_embedding = PathEmbedding(d_node, l_path, d_path)
        self.behav_embedding = BehaviorEmbedding(d_behav, l_path, d_path)
        self.path_interaction = PathInteraction(d_path, n_head, n_encoder, d_feedforward, dropout, activation)
        self.path_aggregation = PathAggregation(aggr_mode)
        self.fraud_detection = nn.Linear(d_path, num_classes)

    def forward(self, g, nodes):
        paths, edge_ids = self.path_sampling(g, nodes)
        path_embed = self.path_embedding(g, paths, edge_ids)
        behav_embed = self.behav_embedding(g, paths)
        path_embed = self.path_interaction(path_embed, behav_embed)
        path_embed = self.path_aggregation(path_embed)
        logits = self.fraud_detection(path_embed)

        return logits


def gpa(**kwargs):
    return GPA(**kwargs)
