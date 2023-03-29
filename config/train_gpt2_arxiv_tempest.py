# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'ppl_nanogpt'
wandb_run_name='gpt2-124m-tempest-arxiv-resumeFrom-wikipedia-2A100'

# these make the total batch size be ~0.16M
# 16 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 163,840
batch_size = 16
block_size = 1024
dataset = 'arxiv'

# it takes ~318 iters to exhaust one epoch of arxiv (of 52,041,540 tokens)
# we run 10 epochs of the training set
# this makes total number of tokens be ~0.5B
max_iters = 3180
lr_decay_iters = 3180

# eval stuff
eval_interval = 400
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

