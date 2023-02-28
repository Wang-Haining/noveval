# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'ppl_nanogpt'
wandb_run_name='gpt2-124m-tempest-2A40'

# these make the total batch size be ~0.16M
# 16 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 163,840
batch_size = 16
block_size = 1024
dataset = 'wikipedia_en'

# it takes 28,352 iters to exhaust one epoch of wikipedia_en (of 4,645,199,244 tokens)
# we run 10 epochs of the training set
# this makes total number of tokens be 46B 
max_iters = 283520
lr_decay_iters = 283520

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

