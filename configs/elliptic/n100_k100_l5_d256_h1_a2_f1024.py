# Copyright (c) QIU Tian. All rights reserved.

import os
import sys

from torch import nn

sys.path.append('configs')
from _base_general_ import *

_ = os.path.split(__file__)[0]
_, dataset = os.path.split(_)

n_bin = 4

model_kwargs = dict(
    d_node=93, d_behav=93,
    n_path=100, k_path=100, l_path=5, d_path=256,
    n_head=1, n_attn=2, d_ffn=1024, dropout=0.1, activation=nn.ReLU,
    aggr_mode='avg'
)

lr = 1e-4

output_dir = f'{output_root}/{dataset}/{os.path.splitext(os.path.basename(__file__))[0]}'
