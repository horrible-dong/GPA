# Copyright (c) QIU Tian. All rights reserved.

# runtime
device = 'cuda'
seed = 42
batch_size = 256
epochs = 200
clip_max_norm = 1.0
eval_interval = 1
num_workers = None  # auto
pin_memory = True
print_freq = 50
amp = True

# dataset
raw_dir = './data/raw'
processed_dir = './data/processed'
dataset = ...

# model
model = 'gpa'

# criterion
criterion = 'default'

# optimizer
optimizer = 'adamw'
lr = 1e-4
weight_decay = 5e-2

# lr_scheduler
scheduler = 'cosine'
warmup_epochs = 0
warmup_steps = 0
warmup_lr = 1e-06
min_lr = 1e-05

# evaluator
evaluator = 'default'

# loading
no_pretrain = True

# saving
save_interval = 10
output_root = './runs'
