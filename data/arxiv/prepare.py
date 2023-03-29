"""
This module prepares the Arxiv corpus [1] corpus for model training and evaluation.
The script is adopted from `karpathy/nanoGPT/data/openwebtext/prepare.py` with some modification.

References:
    Clement, C. B., Bierbaum, M., O'Keeffe, K. P., & Alemi, A. A. (2019). On the Use of ArXiv as a Dataset.
arXiv preprint arXiv:1905.00075.
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)

# verification leads to failure (on huggingface datasets side)
dataset = load_dataset("arxiv_dataset", data_dir='data/arxiv', ignore_verifications=True)
# filter it to get relevant abstracts
# check arxiv filed abbreviations on https://arxiv.org/category_taxonomy
relevant_fields = ['cs.' + f for f in ['AI', 'CL', 'CY', 'HC', 'IR', 'LG', 'SI', 'GL', 'DL']] + ['stats.ML']
dataset = dataset.filter(lambda x: any(f in x['categories'] for f in relevant_fields))
# reserve 'abstract' ('text') feature only, now we have ~200m abstracts in relevant fields
dataset = dataset.remove_columns(['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi', 'report-no',
                                  'categories', 'license', 'update_date']).rename_column('abstract', 'text')

# arxiv corpus by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 218958
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 11525
#     })
# })

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
