"""
This script complements the section on using known-group techniques to justify construct validity in the paper 'A
Novelty Measure for Scholarly Publications Aligned with Peer Review'.
It evaluates the novelty of academic papers from the CLEF PAN Shared Task - Authorship Verification using surprisal
values from a GPT model trained on the English Wikipedia.
"""

__author__ = "hw56@indiana.edu"

import json
import os

import numpy as np
import torch
from scipy.stats import ttest_ind

from model import GPT, GPTConfig
from utils import calculate_surprisal

if __name__ == "__main__":
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = "out_wikipedia_en"
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    # read in CLEF PAN Authorship Verification corpus for authorship verification
    input_file = "resource/clef_pan_av.jsonl"
    papers = []
    with open(input_file, "r") as f:
        for line in f:
            papers.append(json.loads(line))
    papers.sort(key=lambda x: x["id"])

    # annotated high novelty paper
    # 26/83 are deemed novel, but only 25/80 are long enough for surprisal calculation
    # fmt: off
    novel_paper_index = [
        5, 14, 15, 30, 33, 34, 48, 50, 51, 52, 53, 54, 55, 60, 63, 64, 66, 67, 71, 73, 74, 75, 79, 80, 82, 83
    ]
    # fmt: on
    papers_surprisal = {}
    for p in papers:
        try:
            surps, _, _, _, _ = calculate_surprisal(
                text=p["abstract"] + "\n\n" + p["content"],
                model=model,
                context_length=256,
                sequence_length=1024,
                use_all_tokens=False,
                device=device,
                compile_model=True,
            )
            # surps = remove_trailing_zeros(surps)  # ignore -1 padded surprisal scores
            print(f"Paper {p['id']} has an average surprisal of {np.mean(surps):.3}.")
            papers_surprisal.update({p["id"]: surps})
        except ValueError:
            print(f"Paper {p['id']} is too short.")  # too-short papers: 1, 44, 51

    novel_paper_avg_surprisal = []
    normal_paper_avg_surprisal = []
    for idx in range(1, len(papers) + 1):
        if papers_surprisal.get(idx, None):
            avg_surprisal = np.mean(papers_surprisal.get(idx))
            if idx in novel_paper_index:
                novel_paper_avg_surprisal.append(avg_surprisal)
            else:
                normal_paper_avg_surprisal.append(avg_surprisal)

    print(
        ttest_ind(
            novel_paper_avg_surprisal,
            normal_paper_avg_surprisal,
            alternative="greater",
            equal_var=False,
        )
    )
    # TtestResult(statistic=2.6514487979253913, pvalue=0.005007329796631405, df=66.33431945489477)

    # # uncomment below to examine a specific paper
    # from utils import decode
    #
    # # take the 13th paper as an example (it's 0-based index, 13th paper is indexed with 12)
    # surps, tops, ids, ranks, raw = calculate_surprisal(
    #     text=papers[12]["abstract"] + "\n\n" + papers[12]["content"],
    #     model=model,
    #     context_length=256,
    #     sequence_length=1024,
    #     use_all_tokens=False,
    #     device=device,
    #     compile_model=True,
    # )
    # print(list(zip(*[surps, [decode([i]) for i in ids]])))
