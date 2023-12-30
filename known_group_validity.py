"""
todo: improve readability
This script evaluates the novelty of academic papers using surprisal values from a GPT model trained on English
Wikipedia.

This tool is essential for research in computational linguistics and digital text forensics, providing insights into the
 relationship between linguistic unpredictability and academic novelty.
"""

import os
import json
import torch
import numpy as np
from model import GPTConfig, GPT
from scipy.stats import ttest_ind
from utils import calculate_surprisal, remove_trailing_zeros

if __name__ == '__main__':
    # load model
    # device = 'cuda:0'
    device = 'cpu'
    out_dir = 'out_wikipedia_en'
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    # read in noveval corpus for authorship verification
    input_file = 'resource/noveval_content.jsonl'
    papers = []
    with open(input_file, 'r') as f:
        for line in f:
            papers.append(json.loads(line))
    papers.sort(key=lambda x: x['id'])

    # annotated high novelty paper
    # 26/83 are deemed novel, but only 25/80 are long enough for surprisal calculation
    novel_paper_index = [
        5, 14, 15, 30, 33, 34, 48, 50, 51, 52, 53, 54, 55, 60, 63, 64, 66, 67, 71, 73, 74, 75, 79, 80, 82, 83
    ]
    papers_surprisal = {}
    for p in papers:
        try:
            surps, _, _, _, _ = calculate_surprisal(text=p['abstract'] + '\n\n' + p['content'],
                                                    model=model,
                                                    computing_method='long_history',
                                                    device=device,
                                                    sequence_length=1024,
                                                    block_size=1024,
                                                    minimum_context_length=256,
                                                    random_start_pos=False,
                                                    random_state=0,
                                                    compile_model=True)
            surps = remove_trailing_zeros(surps)  # ignore -1 padded surprisal scores
            print(f"{p['id']} has an average surprisal of {np.mean(surps):.3}.")
            papers_surprisal.update({p['id']: surps})
        except ValueError:
            print(f"{p['id']} is too short.")  # too-short papers: 1, 44, 51

    novel_paper_avg_surprisal = []
    normal_paper_avg_surprisal = []
    for idx in range(1, len(papers) + 1):
        if papers_surprisal.get(idx, None):
            avg_surprisal = np.mean(papers_surprisal.get(idx))
            if idx in novel_paper_index:
                novel_paper_avg_surprisal.append(avg_surprisal)
            else:
                normal_paper_avg_surprisal.append(avg_surprisal)

    print(ttest_ind(novel_paper_avg_surprisal, normal_paper_avg_surprisal, alternative='greater', equal_var=False))
    # TtestResult(statistic=2.6514562282286063, pvalue=0.005007229608955063, df=66.33436896164095)

    # examine interest paper with the following code
    # from utils import remove_trailing_zeros, remove_trailing_negative_ones, decode
    # take the 13th paper as an example (it's 0-based index, 13th paper is indexed with 12)
    # surps, tops, ids, ranks, raw = calculate_surprisal(text=papers[12]['abstract'] + '\n\n' + papers[12]['content'],
    #                                         model=model,
    #                                         computing_method='long_history',
    #                                         device=device,
    #                                         sequence_length=1024,
    #                                         block_size=1024,
    #                                         minimum_context_length=256,
    #                                         random_start_pos=False,
    #                                         random_state=0,
    #                                         compile_model=True)
    # print(list(zip(*[remove_trailing_zeros(surps), [decode([i]) for i in remove_trailing_negative_ones(ids)]])))
