# Copyright (c) QIU Tian. All rights reserved.

import os
import sys

from torch import nn

sys.path.append('configs')
from _base_general_ import *

_ = os.path.split(__file__)[0]
_, dataset = os.path.split(_)

batch_size = 2000
n_bin = 32

model_kwargs = dict(
    d_node=10, d_behav=10,
    n_path=200, k_path=100, l_path=30, d_path=64,
    n_head=8, n_attn=2, d_ffn=1024, dropout=0.1, activation=nn.ReLU,
    aggr_mode='avg'
)

output_dir = f'{output_root}/{dataset}/{os.path.splitext(os.path.basename(__file__))[0]}'

print_freq = 400
