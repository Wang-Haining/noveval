import os
import torch
import numpy as np
import pandas as pd
from utils import get_paper_and_score, calculate_perplexity, encode, decode
from model import GPT, GPTConfig

# result_0407_path = 'results_stale_0407/mdl=wikipedia_en-mtd=long_history-mcl=512.csv'
# result_path = 'results/mdl=wikipedia_en-mtd=long_history-mcl=512.csv'
#
# result_0407 = pd.read_csv(result_0407_path)
# result = pd.read_csv(result_path)

# load paper and review score
review_scores = get_paper_and_score(corpus_path="./PeerRead/data/acl_2017/", preserve_ordinal=True)

# args
device = 'cuda:0'
sequence_length = 2048

# load model
ckpt_path = os.path.join('out_wikipedia_en', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

# calculate ppl
ppl = [calculate_perplexity(text=text,
                            model=model,
                            computing_method='long_history',
                            device=device,
                            sequence_length=2048,
                            block_size=1024,
                            minimum_context_length=512,
                            sampling=True,
                            random_state=0,
                            compile_model=True,
                            verbosity=True) for text in review_scores['paper']]
