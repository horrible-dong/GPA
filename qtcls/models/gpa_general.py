# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['GeneralGPA', 'gpa_general']

from torch import nn

from ..modules.gpa_general import PathSampling
from ..modules.gpa_general import PathEmbedding
from ..modules.gpa_general import BehaviorEmbedding
from ..modules.gpa_general import PathInteraction
from ..modules.gpa_general import PathAggregation
from ..modules.gpa_general import FraudDetection


class GeneralGPA(nn.Module):
    """
    A general version for training on any dataset,
    especially for Elliptic, T-Finance, T-Social, YelpChi, Amazon, etc.
    with the meaning of nodes, edges, relations, etc. unknown.
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
        paths = self.path_sampling(g, nodes)
        path_embed = self.path_embedding(g, paths)
        behav_embed = self.behav_embedding(g, paths)
        path_embed = self.path_interaction(path_embed, behav_embed)
        path_embed = self.path_aggregation(path_embed)
        logits = self.fraud_detection(path_embed)

        return logits


def gpa_general(**kwargs):
    return GeneralGPA(**kwargs)
