import os
import torch
import argparse
from model import GPTConfig, GPT

import pandas as pd
from utils import get_paper_and_score
from utils import calculate_surprisal

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate surprisal score from a document using a bespoke GPT")
    parser.add_argument("--model_dir", default="out_wikipedia_en", type=str, help="directory a ckpt is saved")
    parser.add_argument("--device", default="cuda:0", type=str, help="directory a ckpt is saved")
    parser.add_argument("--random_state", default=0, type=int, help="seed")
    parser.add_argument("--sequence_length", default=2048, type=int, help="num of tokens whose ppl will be returned")
    parser.add_argument("--block_size", default=1024, type=int, help="model block_size")
    parser.add_argument("--minimum_context_length", default=512, type=int,
                        help="minimum num of tokens each token's surprisal calculation have to condition on")
    parser.add_argument("--top_k", default=30, type=int, help="number of top candidates to consider at each position")
    parser.add_argument("--add_initial_eot", dest="add_initial_eot", action="store_true",
                        help="add an eot token in the beginning of the text")
    parser.add_argument("--random_start_pos", dest="random_start_pos", action="store_true")
    parser.set_defaults(add_initial_eot=True)
    parser.add_argument("--computing_method", default="long_history",
                        choices=["long_history", "naive"])
    parser.add_argument("--no-compile_model", dest="compile_model", action="store_false",
                        help="do not compile model for efficiency, supported by pytorch 2.0")
    parser.set_defaults(compile_model=True)
    args = parser.parse_args()

    # load paper and review score
    review_scores = get_paper_and_score(corpus_path="./PeerRead/data/acl_2017/", preserve_ordinal=True)

    # load model
    ckpt_path = os.path.join(args.model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    # calculate ppl
    surps, tops, ids, ranks, raw = zip(*[calculate_surprisal(text=text,
                                                             model=model,
                                                             computing_method=args.computing_method,
                                                             device=args.device,
                                                             sequence_length=args.sequence_length,
                                                             block_size=args.block_size,
                                                             minimum_context_length=args.minimum_context_length,
                                                             add_initial_eot=args.add_initial_eot,
                                                             random_start_pos=args.random_start_pos,
                                                             random_state=args.random_state,
                                                             compile_model=args.compile_model) for text in
                                         review_scores['paper']])

    review_scores.update({'surprisals': surps})
    review_scores.update({'top_candidates_at_each_position': tops})
    review_scores.update({'ids': ids})
    review_scores.update({'target_token_rank': ranks})
    review_scores.update({'text': raw})

    df = pd.DataFrame.from_dict(review_scores)
    if args.computing_method == 'long_history':
        df.to_csv(
            f'./results/mdl={args.model_dir[4:]}-mtd={args.computing_method}-mcl={args.minimum_context_length}.csv',
            index=False)
    else:
        df.to_csv(f'./results/mdl={args.model_dir[4:]}-mtd={args.computing_method}.csv',
                  index=False)
