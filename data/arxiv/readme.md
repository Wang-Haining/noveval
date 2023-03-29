
## arxiv dataset

### get raw data before running `prepare.py`

Visit https://www.kaggle.com/datasets/Cornell-University/arxiv and download the dataset (~1GB in .zip). We leave the 
process manual because it requires registration (and managing credential).
Save the downloaded `archive.zip` file to `./data/arxiv`. Then run

```bash
unzip data/arxiv/archive.zip -d data/arxiv/ 
```
We should get `arxiv-metadata-oai-snapshot.json` saved in the same directory. 

Then run `python data/arxiv/prepare.py`, we can get:

- train.bin is ~8.7GB, val.bin ~4.3MB
- train has ~4.6B tokens (4,645,199,244)
- val has ~2.2M tokens (2,231,052)

this came from 6,458,670 documents in total.

references:

- https://huggingface.co/datasets/wikipedia
