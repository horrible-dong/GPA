# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['GPA', 'gpa']

from torch import nn

from ..modules.gpa import PathSampling
from ..modules.gpa import PathEmbedding
from ..modules.gpa import BehaviorEmbedding
from ..modules.gpa import PathInteraction
from ..modules.gpa import PathAggregation
from ..modules.gpa import FraudDetection


class GPA(nn.Module):
    """
    A specific version for training on the G-Internet dataset
    with the meaning of nodes, edges, relations, etc. known.
    """

    def __init__(self,
                 d_node, d_behav,
                 n_path, k_path, l_path, d_path,
                 n_head, n_attn, d_ffn, dropout, activation,
                 aggr_mode,
                 num_classes):
        super().__init__()
        self.path_sampling = PathSampling(n_path, k_path, l_path)
        self.path_embedding = PathEmbedding(d_node, l_path, d_path)
        self.behav_embedding = BehaviorEmbedding(d_behav, l_path, d_path)
        self.path_interaction = PathInteraction(d_path, n_head, n_attn, d_ffn, dropout, activation)
        self.path_aggregation = PathAggregation(aggr_mode)
        self.fraud_detection = FraudDetection(d_path, num_classes)

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
