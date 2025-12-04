# Copyright (c) QIU Tian. All rights reserved.

import os
import sys

from torch import nn

sys.path.append('configs')
from _base_ import *

_ = os.path.split(__file__)[0]
_, dataset = os.path.split(_)

batch_size = 2048

model_kwargs = dict(
    d_node=106, d_behav=3,
    n_path=200, k_path=150, l_path=20, d_path=128,
    n_head=8, n_attn=2, d_ffn=512, dropout=0.1, activation=nn.ReLU,
    aggr_mode='avg'
)

output_dir = f'{output_root}/{dataset}/{os.path.splitext(os.path.basename(__file__))[0]}'
