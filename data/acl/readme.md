
## arXiv dataset

### get raw data before running `prepare.py`

Visit https://github.com/shauryr/ACL-anthology-corpusand download the dataset (~490MB in .parquet). 


Then run `python data/acl/prepare.py`, we can get:

- train.bin is ~ 316MB, val.bin ~17MB  # todo
- train has 165,309,909 tokens
- val has 8,910,174 tokens

this came from 25,634 abstracts from relevant fields published before 2016.

references:

- https://github.com/shauryr/ACL-anthology-corpus
