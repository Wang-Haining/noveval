# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'ppl_nanogpt'
wandb_run_name = 'gpt2-124m-tempest-1A100-arxiv'
init_from = 'resume'

# these make the total batch size be ~0.08M
# 16 batch size * 1024 block size * 5 gradaccum * 1 GPU = 81,920
batch_size = 16
block_size = 1024
dataset = 'arxiv'

# it takes ~60 iters to exhaust one epoch of arxiv (of 4,944,468 tokens)
# we run 10 epochs of training
# this makes total number of tokens be ~50m
max_iters = 141000 + 600  # 141000 is the total steps of the pretrained gpt2

# eval stuff
eval_interval = 5
eval_iters = 20
log_interval = 1

# finetune at constant LR
learning_rate = 3e-5  # half of min_LR of the pretraining
decay_lr = False

# no compile, as it is not stable
compile = False
