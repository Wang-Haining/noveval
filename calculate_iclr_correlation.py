import os
import json

import torch
import tiktoken
import numpy as np
import pandas as pd
from model import GPTConfig, GPT
from itertools import chain
import math

from calculate_ppl import calculate_ppl
from datetime import datetime

corpus_path = "./PeerRead/data/iclr_2017/"

paths = [corpus_path + s for s in ['train', 'dev', 'test']]
paper_paths = [p.path for p in chain(*[os.scandir(os.path.join(dir, 'parsed_pdfs')) for dir in paths])]
review_paths = [p.path for p in chain(*[os.scandir(os.path.join(dir, 'reviews')) for dir in paths])]

papers = []
for f in paper_paths:
    paper = json.load(open(f, 'r'))
    papers.append(paper)

text_dict = [{int(paper['name'].split('.pdf')[0]): "\n\n".join([d['text'] for d in paper['metadata']['sections']])} for
             paper in papers if int(paper['name'].split('.pdf')[0]) != 621]

aspects = [
    'IMPACT',
    'SUBSTANCE',
    'APPROPRIATENESS',
    'SOUNDNESS_CORRECTNESS',
    'ORIGINALITY', 'RECOMMENDATION', 'CLARITY',
    'REVIEWER_CONFIDENCE',
    'RECOMMENDATION_UNOFFICIAL']

reviews = []
for f in review_paths:
    review = json.load(open(f, 'r'))
    reviews.append(review)


def round_up(mean_score: float):
    return int(math.ceil(mean_score - 0.5))


review_aspect_scores = []
for review in reviews:  # iterate over all records
    paper_review_tmp = []
    for review_dict in review['reviews']:  # review['reviews'] contains ~10 reviews for a paper
        if "AnonReviewer" in review_dict.get('OTHER_KEYS', ''):  # 'OTHER_KEYS' stores the origin of the review
            review_aspects = {'id': int(review.get('id'))}  # get every aspect of a reviewer
            for aspect in aspects + ['OTHER_KEYS']:
                if aspect != 'DATE':
                    review_aspects.update({aspect: review_dict.get(aspect, None)})
                else:
                    try:
                        review_aspects.update({aspect: datetime.strptime(review_dict.get(aspect, None),
                                                                         "%d %b %Y")})  # may not match format or get None
                    except (ValueError, TypeError) as e:
                        print(e)
            paper_review_tmp.append(review_aspects)  # stores date, id, and aspects of relevant reviewers (duplicates exsit for now)
    # merge score of each reviewer; take the latest if scores of a specific reviewer disagree in history
    reviewer_final_scores = []  # keys: ['id', 'reviewer'] + aspects
    reviewers = sorted(set([r['OTHER_KEYS'] for r in paper_review_tmp]))  # find out unique reviewers
    for reviewer in reviewers:
        reviewer_final_score = {'reviewer': reviewer, 'id': int(review.get('id'))}  #
        for _aspect in aspects:
            # select int scores sorted by time
            aspect_score_tmp = [{'DATE': d.get('DATE'), _aspect: d[_aspect]} for d in paper_review_tmp if
                                (d['OTHER_KEYS'] == reviewer and isinstance(d[_aspect], int))]
            # print(aspect_score_tmp)
            if len(aspect_score_tmp) == 0:  # a reviewer never rates a specific aspect
                reviewer_final_score.update({_aspect: None})
            elif len(aspect_score_tmp) == 1:  # only one aspect rating from a reviewer
                reviewer_final_score.update(aspect_score_tmp[0])
            else:  # multiple ratings from a reviewer; solve conflict
                try:
                    sorted_aspect_score_tmp = sorted(aspect_score_tmp, key=lambda x: x['DATE'])
                    reviewer_final_score.update(sorted_aspect_score_tmp[-1])
                except (ValueError, TypeError):
                    if all(aspect_score_tmp[0][_aspect] == d[_aspect] for d in aspect_score_tmp):
                        reviewer_final_score.update(aspect_score_tmp[0])
                    else:
                        reviewer_final_score.update(
                            {'DATE': None, _aspect: round_up(np.mean([d[_aspect] for d in aspect_score_tmp]))})
        reviewer_final_scores.append(reviewer_final_score)
    review_aspect_scores.extend(reviewer_final_scores)

# paper 621 does not have corresponding sections: https://github.com/allenai/PeerRead/blob/master/data/iclr_2017/dev/parsed_pdfs/621.pdf.json
# we have 426 papers in total
assert set([d['id'] for d in review_aspect_scores]) - set([list(d.keys())[0] for d in text_dict]) == {621}

review_scores = []
for paper_id in [list(d.keys())[0] for d in text_dict]:
    paper_review = {'id': paper_id}
    for aspect in aspects:
        scores = [r[aspect] for r in review_aspect_scores if (r['id'] == paper_id and r[aspect] is not None)]
        if len(scores) >= 1:
            paper_review.update({aspect.lower(): round_up(np.mean(scores))})
        else:  # no ratings
            paper_review.update({aspect.lower(): None})
    paper_review.update({"paper": [list(d.values())[0] for d in text_dict if list(d.keys())[0] == paper_id][0]})
    review_scores.append(paper_review)


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
for d in review_scores:
    try:
        ppl = calculate_ppl(text=d['paper'],
                            model=model,
                            tokenizer=encode,
                            device=device,
                            sequence_length=2048,
                            block_size=1024,
                            sliding_window_length=512,
                            random_state=0,
                            compile_model=True)
    except RuntimeError:
        ppl = None
    d.update({'ppl': ppl})

df = pd.DataFrame(review_scores)
df.to_csv('./results/iclr_corpus_ordinal.csv', index=False)
