wandb_log = True
wandb_project = 'noveval'
wandb_run_name = 'gpt2-124m-br200-4A100-acl'
init_from = 'resume'

# these make the total batch size be ~0.33M
# 16 batch size * 1024 block size * 5 gradaccum * 4 GPU = 327,680
batch_size = 16
block_size = 1024
dataset = 'acl'
gradient_accumulation_steps = 5

# it takes ~561 iters to exhaust one epoch of acl (of 183,981,198 tokens)
# we run ~11 epochs of finetuning
# to get a total number of tokens of ~2B
max_iters = 141000 + 6000  # we resume from the pretrained gpt2 (141000 iters)

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# finetune at constant LR
learning_rate = 6e-5  #  min_LR of the pretraining
decay_lr = False

# no compile, as it is not stable
compile = False

# adamw eps
eps = 1e-5
