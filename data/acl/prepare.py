"""
This module prepares a subset of the acl anthology corpus (https://github.com/shauryr/ACL-anthology-corpus) for
finetuning a GPT-2 model.

The script is adopted from `karpathy/nanoGPT/data/openwebtext/prepare.py` with some modification.
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import pyarrow.parquet as pq
from datasets import Dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
dataset = Dataset(pq.read_table('data/acl/acl-publication-info.74k.parquet', memory_map=True))

# filter it to get relevant abstracts

# dataset = dataset.filter(lambda x: any(f in x['categories'] for f in relevant_fields))
dataset = dataset.filter(lambda x: int(x['year']) <= 2015)
# now we have 25634 abstracts in relevant fields updated before 2016
# reserve 'abstract' ('text') feature only
dataset = dataset.remove_columns(['acl_id', 'abstract', 'corpus_paper_id', 'pdf_hash',
                                  'numcitedby', 'url', 'publisher', 'address', 'year', 'month', 'booktitle',
                                  'author', 'title', 'pages', 'doi', 'number', 'volume', 'journal', 'editor',
                                  'isbn', '__index_level_0__']).rename_column('full_text', 'text')

# arxiv corpus by default only contains the 'train' split, so create a test split
split_dataset = dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 39104
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 2059
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
