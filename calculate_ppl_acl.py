import os
import torch
import argparse
from model import GPTConfig, GPT

import pandas as pd
from utils import get_paper_and_score
from utils import calculate_perplexity, calculate_type_token_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate perplexity score with control")
    parser.add_argument("--out_dir", default="out_wikipedia_en", type=str, help="directory a ckpt is saved")
    parser.add_argument("--device", default="cuda:0", type=str, help="directory a ckpt is saved")
    parser.add_argument("--random_state", default=0, type=int, help="seed")
    parser.add_argument("--ppl_computing_method", default="well_contextualized",
                        choices=["long_history", "naive"])
    parser.add_argument("--ignore_function_words", action="store_true",
                        help="Ignore function words when accumulating loss")
    parser.add_argument("--no-ignore_function_words", dest="ignore_function_words", action="store_false")
    parser.add_argument("--model_compile", action="store_true",
                        help="Compile model for efficiency, supported by pytorch 2.0")
    parser.add_argument("--no-model_compile", dest="model_compile", action="store_false")
    parser.set_defaults(model_compile=True)
    args = parser.parse_args()

    # load paper and review score
    review_scores = get_paper_and_score(corpus_path="./PeerRead/data/acl_2017/", preserve_ordinal=True)

    # load model
    out_dir = args.out_dir
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    # calculate ppl
    ppl = [calculate_perplexity(text=text,
                                model=model,
                                ppl_computing_method=args.ppl_computing_method,
                                ignore_function_words=args.ignore_function_words,
                                device=args.device,
                                sequence_length=2048,
                                block_size=1024,
                                sliding_window_length=512,
                                random_state=args.random_state,
                                compile_model=True) for text in review_scores['paper']]

    review_scores.update({'ppl': ppl})

    ttr = [calculate_type_token_ratio(text=text,
                                      sequence_length=2048,
                                      random_state=args.random_state) for text in review_scores['paper']]
    review_scores.update({'ttr': ttr})

    df = pd.DataFrame.from_dict(review_scores)
    df.to_csv(f'./results/model={args.out_dir[4:]}method={args.ppl_computing_method}-ignore_func_words={args.ignore_function_words}.csv',
              index=False)
