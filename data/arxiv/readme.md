
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

- train.bin is ~100MB, val.bin ~5.3MB
- train has ~52MB tokens (52,041,540)
- val has ~2.7M tokens (2,752,404)

this came from 230,483 STEM abstracts in total.

references:

- https://huggingface.co/datasets/wikipedia
