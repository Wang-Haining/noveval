import os
import torch
import tiktoken
from model import GPTConfig, GPT

import pandas as pd
from utils import get_paper_and_score
from calculate_ppl import calculate_ppl

# load paper and review score
review_scores = get_paper_and_score(corpus_path="./PeerRead/data/acl_2017/", preserve_ordinal=True)

# load model
device = 'cpu'
out_dir = 'out_wikipedia_en'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

# load tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

# calculate ppl
ppl = [calculate_ppl(text=text,
                     model=model,
                     tokenizer=encode,
                     device=device,
                     sequence_length=2048,
                     block_size=1024,
                     sliding_window_length=512,
                     random_state=0,
                     compile_model=True) for text in review_scores['paper']]

review_scores.update({'ppl': ppl})

df = pd.DataFrame.from_dict(review_scores)
df.to_csv('./results/acl_corpus_ordinal.csv', index=False)
