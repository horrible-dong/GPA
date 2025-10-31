# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['preprocess_g_internet']

import os

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import train_test_split

from ..utils.misc import index_to_mask
from ..utils.os import makedirs


def preprocess_g_internet(raw_dir, processed_dir, processed_file):
    makedirs(processed_dir, exist_ok=True)

    g = load_graphs(os.path.join(raw_dir, 'g_internet'))[0][0]

    src_user_to_web, dst_user_to_web = g.edges(etype=('user', 'visits', 'web'))
    dst_user_to_web = dst_user_to_web + g.num_nodes('user')
    src_user_to_user, dst_user_to_user = g.edges(etype=('user', 'to', 'user'))
    eweights_user_to_web = g.edges['visits'].data['weight']
    num_webs = g.num_nodes('web')

    user_feats = g.nodes['user'].data['feat']
    web_feats = g.nodes['web'].data['feat']
    user_labels = g.nodes['user'].data['label']
    web_labels = torch.full([num_webs], -1)

    feats = torch.cat([user_feats, web_feats])
    labels = torch.cat([user_labels, web_labels])
    src = torch.cat([src_user_to_web, dst_user_to_web, src_user_to_user])
    dst = torch.cat([dst_user_to_web, src_user_to_web, dst_user_to_user])
    eweights = torch.cat([eweights_user_to_web, eweights_user_to_web, torch.full_like(src_user_to_user, 1)])

    user_behaviors = []
    counts_user_to_web = src_user_to_web.unique(return_counts=True)[1].tolist()
    counts_user_to_user = [0] * len(user_labels)
    for src_user in src_user_to_user.tolist():
        counts_user_to_user[src_user] += 1
    for w, u in zip(eweights_user_to_web.split(counts_user_to_web), counts_user_to_user):
        user_behaviors.append([len(w), torch.sum(w).item(), u])
    user_behaviors = torch.tensor(user_behaviors, dtype=torch.float32)
    user_behaviors = user_behaviors / user_behaviors.max(0)[0]
    user_behaviors = (user_behaviors - user_behaviors.mean(0)) / user_behaviors.std(0)
    behaviors = torch.cat([user_behaviors, torch.zeros(len(web_labels), user_behaviors.shape[1])])

    graph = dgl.graph((src, dst))

    graph.ndata['feat'] = feats
    graph.ndata['label'] = labels
    graph.edata['weight'] = eweights

    graph.ndata['centrality'] = graph.in_degrees()
    graph.ndata['behav'] = behaviors

    num_nodes = graph.num_nodes()
    y = user_labels

    train_idx, test_idx, y_train, y_test = train_test_split(range(len(y)), y, stratify=y, train_size=0.4,
                                                            random_state=0, shuffle=True)
    val_idx, test_idx, y_val, y_test = train_test_split(test_idx, y_test, stratify=y_test, test_size=0.67,
                                                        random_state=0, shuffle=True)

    train_idx = torch.LongTensor(np.sort(train_idx))
    val_idx = torch.LongTensor(np.sort(val_idx))
    test_idx = torch.LongTensor(np.sort(test_idx))

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    processed_path = os.path.join(processed_dir, processed_file)
    save_graphs(processed_path, graph)
    os.chmod(processed_path, 0o777)
