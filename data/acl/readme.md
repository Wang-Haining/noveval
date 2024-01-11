
## acl ocl dataset

### get raw data before running `prepare.py`

Visit https://github.com/shauryr/ACL-anthology-corpusand download the dataset (~490MB in .parquet). 


Then run `python data/acl/prepare.py`, we can get:

- train.bin is ~351MB, val.bin ~18MB  
- train has 183,981,198 tokens
- val has 9,426,326 tokens

this came from 39,663 acl anthology papers published before 2017.

references:

- https://github.com/shauryr/ACL-anthology-corpus
