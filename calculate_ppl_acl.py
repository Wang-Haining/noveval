import os
import torch
import argparse
from model import GPTConfig, GPT

import pandas as pd
from utils import get_paper_and_score
from utils import calculate_perplexity, calculate_type_token_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate perplexity score from a document")
    parser.add_argument("--model_dir", default="out_wikipedia_en", type=str, help="directory a ckpt is saved")
    parser.add_argument("--device", default="cuda:0", type=str, help="directory a ckpt is saved")
    parser.add_argument("--random_state", default=0, type=int, help="seed")
    parser.add_argument("--sequence_length", default=2048, type=int, help="num of tokens whose ppl will be returned")
    parser.add_argument("--block_size", default=1024, type=int, help="model block_size")
    parser.add_argument("--minimum_context_length", default=512, type=int,
                        help="minimum num of tokens each token's ppl "
                             "calculation have to condition on")
    parser.add_argument("--add_initial_eot", action="store_true",
                        help="add an eot token in the beginning of the text")
    parser.add_argument("--no-add_initial_eot", dest="add_initial_eot", action="store_false")
    parser.set_defaults(add_initial_eot=True)
    parser.add_argument("--computing_method", default="long_history",
                        choices=["long_history", "naive"])
    parser.add_argument("--model_compile", action="store_true",
                        help="compile model for efficiency, supported by pytorch 2.0")
    parser.add_argument("--no-model_compile", dest="model_compile", action="store_false")
    parser.set_defaults(model_compile=True)
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
    ppl, cross_entropy, text = zip(*[calculate_perplexity(text=text,
                                                          model=model,
                                                          computing_method=args.computing_method,
                                                          device=args.device,
                                                          sequence_length=args.sequence_length,
                                                          block_size=args.block_size,
                                                          minimum_context_length=args.minimum_context_length,
                                                          add_initial_eot=args.add_initial_eot,
                                                          sampling=True,
                                                          random_state=args.random_state,
                                                          compile_model=True,
                                                          verbosity=True) for text in review_scores['paper']])

    review_scores.update({'ppl': ppl})
    review_scores.update({'cross_entropy': cross_entropy})
    review_scores.update({'text': text})

    # reuse `text` sampled in ppl calculation for ttr
    ttr = [calculate_type_token_ratio(text=t,
                                      sampling=False,
                                      sequence_length=args.sequence_length) for t in text]
    review_scores.update({'ttr': ttr})

    df = pd.DataFrame.from_dict(review_scores)
    if args.computing_method == 'long_history':
        df.to_csv(
            f'./results/mdl={args.model_dir[4:]}-mtd={args.computing_method}-mcl={args.minimum_context_length}.csv',
            index=False)
    else:
        df.to_csv(f'./results/mdl={args.model_dir[4:]}-mtd={args.computing_method}.csv',
                  index=False)
