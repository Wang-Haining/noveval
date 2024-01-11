wandb_log = True
wandb_project = 'noveval'
wandb_run_name = 'gpt2-124m-br200-4A100-wikipedia_en'

# these make the total batch size ~0.33M
# 16 batch size * 1024 block size * 5 gradaccum * 4 GPUs = 327,680
batch_size = 16
block_size = 1024
dataset = 'wikipedia_en'
gradient_accumulation_steps = 5

# it takes 14,176 iters to exhaust one epoch of wikipedia_en
# we run 10 epochs of training
# this makes total number of tokens of 46B
max_iters = 141000
lr_decay_iters = 141000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# no compile, as it is not stable
compile = False

# adamw eps
eps = 1e-5