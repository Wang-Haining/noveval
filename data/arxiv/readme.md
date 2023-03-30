
## arXiv dataset

### get raw data before running `prepare.py`

Visit https://www.kaggle.com/datasets/Cornell-University/arxiv and download the dataset (~1GB in .zip). We leave the 
process manual because it requires registration (and managing credential).
Save the downloaded `archive.zip` file to `./data/arxiv`. Then run

```bash
unzip data/arxiv/archive.zip -d data/arxiv/ 
```
We should get `arxiv-metadata-oai-snapshot.json` saved in the same directory. 

Then run `python data/arxiv/prepare.py`, we can get:

- train.bin is ~ MB, val.bin ~5.3MB
- train has 4,944,468 tokens
- val has 260,797 tokens

this came from 25,634 abstracts from relevant fields published before 2016.

references:

- Clement, C. B., Bierbaum, M., O'Keeffe, K. P., & Alemi, A. A. (2019). On the Use of ArXiv as a Dataset.
arXiv preprint arXiv:1905.00075.
