# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'ppl_nanogpt'
wandb_run_name='gpt2-124m-br200-2A100-arxiv'
init_from = 'resume'

# these make the total batch size be ~0.08M
# 16 batch size * 1024 block size * 5 gradaccum * 2 GPU = 163,480
batch_size = 16
block_size = 1024
dataset = 'arxiv'

# it takes ~318 iters to exhaust one epoch of arxiv (of 52,041,540 tokens)
# we run 10 epochs of training
# this makes total number of tokens be .5B
max_iters = 141000 + 3500  # 141000 is the total steps of the pretrained gpt2

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# finetune at constant LR
learning_rate = 6e-5  # min_LR of the previous run
decay_lr = False

# no compile, as it is not stable
compile = False
