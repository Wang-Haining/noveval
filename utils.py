import os
import re
import math
import json
import numpy as np
from itertools import chain


def get_paper_and_score(corpus_path: str = "./PeerRead/data/acl_2017/", preserve_ordinal=True):

    aspects = [
        'SUBSTANCE', 'APPROPRIATENESS', 'SOUNDNESS_CORRECTNESS', 'ORIGINALITY', 'RECOMMENDATION', 'CLARITY',
        'REVIEWER_CONFIDENCE']

    def round_up(mean_score: float):
        return int(math.ceil(mean_score - 0.5))

    def get_review_aspect_score(reviews, aspect, preserve_ordinal):
        if not preserve_ordinal:
            aspect_dict = [{int(r['id']): np.mean([int(d[aspect]) for d in r['reviews']])} for r in reviews]
        else:
            aspect_dict = [{int(r['id']): round_up(np.mean([int(d[aspect]) for d in r['reviews']]))} for r in reviews]
        aspect_sorted_list = [list(d.values())[0] for d in sorted(aspect_dict, key=lambda x: list(x.keys())[0])]
        aspect_index_list = [list(d.keys())[0] for d in sorted(aspect_dict, key=lambda x: list(x.keys())[0])]
        return aspect_sorted_list, aspect_index_list


    paths = [corpus_path + s for s in ['train', 'dev', 'test']]
    paper_paths = [p.path for p in chain(*[os.scandir(os.path.join(dir, 'parsed_pdfs')) for dir in paths])]
    review_paths = [p.path for p in chain(*[os.scandir(os.path.join(dir, 'reviews')) for dir in paths])]

    # get papers
    papers = []
    for f in paper_paths:
        paper = json.load(open(f, 'r'))
        papers.append(paper)

    text_dict = [{int(paper['name'].split('.pdf')[0]): re.sub(r"\n\d+", "", "\n\n".join(
        [d['text'] for d in paper['metadata']['sections']])).replace("1 000\n\n", '')} for paper in papers]
    text_sorted_list = [list(d.values())[0] for d in sorted(text_dict, key=lambda x: list(x.keys())[0])]
    text_check_list = [list(d.keys())[0] for d in sorted(text_dict, key=lambda x: list(x.keys())[0])]

    # get review scores
    reviews = []
    for f in review_paths:
        review = json.load(open(f, 'r'))
        reviews.append(review)

    review_scores = {}
    review_checks = {}
    for aspect in aspects:
        score, check = get_review_aspect_score(reviews, aspect, preserve_ordinal)
        review_scores.update({aspect.lower(): score})
        review_checks.update({aspect.lower(): check})

    if not all(v == text_check_list for v in list(review_checks.values())):
        raise AttributeError("Expect aligned indices between paper and review aspect scores.")

    # merge text and score
    review_scores.update({'paper': text_sorted_list})

    return review_scores
