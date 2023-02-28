# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'ppl_nanogpt'
wandb_run_name='gpt2-124m-br200-4A100'

# these make the total batch size be ~0.33M
# 16 batch size * 1024 block size * 5 gradaccum * 4 GPUs = 327,680
batch_size = 16
block_size = 1024
dataset = 'wikipedia_en'

# it takes 14,176 iters to exhaust one epoch of wikipedia_en
# we run 10 epochs of training
# this makes total number of tokens be 46B
max_iters = 141760
lr_decay_iters = 141760

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# no compile, as it is not stable
compile = False
