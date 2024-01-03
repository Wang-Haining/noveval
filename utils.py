"""
This script provides utilities useful in producing the findings in the paper 'A Novelty Measure for Scholarly
Publications Aligned with Peer Review'.
"""

import os
import re
import math
import json
import torch
import warnings
import tiktoken
import matplotlib
import numpy as np
from model import GPT
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional


__author__ = "hw56@indiana.edu"


def prepare_acl2017_corpus(corpus_path: str = "./PeerRead/data/acl_2017/",
                           preserve_ordinal: bool = True) -> Dict[str, List]:
    """
    Prepare ACL2017 papers and novelty ratings from ACL2017 subset of PeerRead.

    Args:
        corpus_path: the root path of the ACL2017 corpus. Defaults to "./PeerRead/data/acl_2017/".
        preserve_ordinal: if True, round up mean scores for preserving ordinal scale. Defaults to True.

    Returns:
        a dictionary containing sorted lists of paper texts, titles, and originality scores.

    Raises:
        AttributeError: if the indices between paper texts and review scores are not aligned.

    References:
        - PeerRead: https://arxiv.org/abs/1804.09635
    """

    def round_up(mean_score: float) -> int:
        """Round up mean score to the nearest whole number for ordinal preservation."""
        return int(math.ceil(mean_score - 0.5))

    def get_review_aspect_score(reviews: List[Dict],
                                aspect: str,
                                preserve_ordinal: bool) -> Tuple[List[int], List[int]]:
        """
        Extract and process aspect scores from reviews.

        Args:
            reviews: list of review data.
            aspect: the aspect to extract scores for.
            preserve_ordinal: if True, round up mean scores for preserving ordinal scale.

        Returns:
            sorted lists of aspect scores and corresponding paper indices.
        """
        if not preserve_ordinal:
            aspect_dict = [{int(r['id']): np.mean([int(d[aspect]) for d in r['reviews']])} for r in reviews]
        else:
            aspect_dict = [{int(r['id']): round_up(np.mean([int(d[aspect]) for d in r['reviews']]))} for r in reviews]
        aspect_sorted_list = [list(d.values())[0] for d in sorted(aspect_dict, key=lambda x: list(x.keys())[0])]
        aspect_index_list = [list(d.keys())[0] for d in sorted(aspect_dict, key=lambda x: list(x.keys())[0])]
        return aspect_sorted_list, aspect_index_list

    # define paths for paper and review data
    split_paths = [os.path.join(corpus_path, s) for s in ['train', 'dev', 'test']]
    paper_paths = [p.path for s in split_paths for p in os.scandir(os.path.join(s, 'parsed_pdfs')) if
                   p.path.endswith('.json')]
    review_paths = [p.path for s in split_paths for p in os.scandir(os.path.join(s, 'reviews')) if
                    p.path.endswith('.json')]

    # process papers
    papers = [json.load(open(f, 'r')) for f in paper_paths]
    text_dict = [{int(paper['name'].split('.pdf')[0]): "\n\n".join(
        [d['text'] for d in paper['metadata']['sections']])} for paper in papers]
    title_dict = [{int(paper['name'].split('.pdf')[0]): paper['metadata']['title']} for paper in papers]
    text_sorted_list = [list(d.values())[0] for d in sorted(text_dict, key=lambda x: list(x.keys())[0])]
    text_check_list = [list(d.keys())[0] for d in sorted(text_dict, key=lambda x: list(x.keys())[0])]
    title_sorted_list = [list(d.values())[0] for d in sorted(title_dict, key=lambda x: list(x.keys())[0])]

    # process reviews
    reviews = [json.load(open(f, 'r')) for f in review_paths]
    review_scores = {}
    review_checks = {}
    score, check = get_review_aspect_score(reviews, 'ORIGINALITY', preserve_ordinal)
    review_scores.update({'originality': score})
    review_checks.update({'originality': check})

    # check if paper and review indices are aligned
    if not all(v == text_check_list for v in list(review_checks.values())):
        raise AttributeError("Expect aligned indices between paper and review aspect scores.")

    review_scores.update({'paper': text_sorted_list, 'title': title_sorted_list})

    return review_scores


def encode(text: str) -> List[int]:
    """
    Encode a document into its corresponding ids found in `titoken`'s encoding.
    """
    enc = tiktoken.get_encoding("gpt2")

    return enc.encode(text, allowed_special={"<|endoftext|>"})


def decode(ids: List[int]) -> str:
    """
    Decode ids into a document.
    """
    enc = tiktoken.get_encoding("gpt2")

    return enc.decode(ids)


def clean_up_artifacts(text_from_parsed_pdf: str) -> str:
    """
    Clean up obvious artifacts seen in the parsed pdfs.

    This function performs the following operations:
    - Removes occurrences of newlines followed by a number (e.g., "\n123").
    - Removes specific artifact patterns like "1 000\n\n" and "1 000\n".
    - Removes long sequences of quotation marks.

    Args:
        text extracted from a parsed PDF.

    Returns:
        cleaned-up text.
    """
    newline_number_pattern = re.compile(r"\n\d+")
    one_thousand_pattern = re.compile(r"1 000\n\n?")
    long_quotes_pattern = re.compile(r'\"{10,}')

    text = newline_number_pattern.sub("", text_from_parsed_pdf)
    text = one_thousand_pattern.sub("", text)
    text = long_quotes_pattern.sub("", text)

    return text


def get_top_choices_and_ranks(logits: torch.Tensor,
                              y: List[int],
                              top_k: int = 30) -> Tuple[List[List[Tuple[str, int, float, float]]], List[int]]:
    """
    Computes the probabilities from the logits and identifies the top 'k' choices for each token. It returns the top
    choices along with their ranks in the sequence. Each choice includes the token, its index, surprisal score, and
    probability.

    Args:
        logits: the logits output from a language model.
        y: a list of token indices representing the actual sequence for comparison.
        top_k: top_k: number of top token choices to consider at each position.

    Returns:
        a tuple containing two elements:
            - a list of lists, each containing tuples for the top 'k' choices per token. Each tuple consists of the
            token (str), its index (int), surprisal score (float), and probability (float).
            - a list of integers representing the rank of each actual token in 'y' within the predicted probabilities.
    """
    logits_np = logits.numpy()
    e_x = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
    probs_np = e_x / e_x.sum(axis=-1, keepdims=True)

    top_choices = []
    ranks = []
    for i, token_probs in enumerate(probs_np):
        # ensure top_k does not exceed the length of token_probs
        current_top_k = min(top_k, token_probs.size)

        # get the top k indices and their corresponding probabilities
        top_indices = np.argpartition(token_probs, -current_top_k)[-current_top_k:]
        top_probs = token_probs[top_indices]
        sorted_indices = top_indices[np.argsort(-top_probs)]

        token_choices = []
        for idx in sorted_indices:
            token = decode([idx])
            token_prob = token_probs[idx]
            token_surprisal = -np.log2(token_prob + 1e-9)
            token_choices.append((token, idx, token_surprisal, token_prob))
        top_choices.append(token_choices)

        # get the rank of the current token in y
        current_token = y[i]
        sorted_probs = np.sort(token_probs)[::-1]  # Sort probabilities in descending order
        # find the rank of the current token's probability
        current_token_rank = np.where(sorted_probs == token_probs[current_token])[0][0] + 1
        ranks.append(current_token_rank)

    return top_choices, ranks


@torch.no_grad()
def calculate_surprisal(text: str,
                        model: GPT,
                        context_length: int = 512,
                        sequence_length: int = 2048,
                        use_all_tokens: bool = False,
                        device: str = 'cpu',
                        top_k: int = 30,
                        block_size: int = 1024,
                        random_state: Optional[int] = None,
                        compile_model: bool = True) -> [np.float64 | Tuple[np.float64, List[np.float64], list, str]]:
    """
    Calculate the surprisal in bits of a continuous sequence of tokens extracted from a given text. The function can
    operate over a specified number of tokens or the entire text if 'use_all_tokens' is True.

    When 'use_all_tokens' is False, the function selects a segment of text based on 'sequence_length' and
    'context_length'. For 'random_start' as True, this segment is chosen randomly. When 'use_all_tokens' is
    True, it processes the entire text, ignoring the 'sequence_length'. Note that the very last tokens, which may not
    naturally fit into a single forward pass, will not be used.

    Args:
        text: an English string, ideally longer than 2,000 words for stable surprisal estimation.
        model: a nano-GPT language model for calculation. Note, the vocabulary size is fixed to 50304.
        context_length: Number of preceding tokens used as context. Defaults to 512. A shorter context may lead
            to less stable surprisal scores. In extreme cases, setting this to 0 corresponds to the loss in each forward
             pass. However, excessively long contexts can result in increased computational demands.
        sequence_length: length of tokens to compute surprisal for. Ignored if 'use_all_tokens' is True. No smaller than
            a `block_size` (1024). Defaults to 2048.
        use_all_tokens: if True, the entire text is processed, regardless of 'sequence_length'. Defaults to False.
        top_k: number of top token choices to consider at each position. Defaults to 30.
        block_size: maximum sequence length the model is designed to handle. Defaults to 1024.
        device: computation device ('cpu', 'cuda:0', etc.). Defaults to 'cpu'.
        random_state: random state for reproducibility when choosing a random start position. Defaults to None.
        compile_model: if True, compiles the PyTorch model (requires PyTorch 2.0). Defaults to False.

    Returns:
        a tuple containing the 2-based surprisal at each position, detailed information about the top candidate tokens
        at each position, token IDs, their rankings, and the portion of text used for computation.
    """
    # sanity check
    if sequence_length < block_size:  # 1024
        raise ValueError(f'Expect `sequence_length` longer than 1024, but got {sequence_length}.')
    # prepare model
    if compile_model:
        model = torch.compile(model)
    model.to(device)
    model.eval()

    # prepare data
    # include the eot token `<|endoftext|>` in the very beginning for a warm start
    data = np.array(encode("<|endoftext|>" + text))  # note "<|endoftext|>" id 50256

    # sanity check
    data_length = len(data)
    if data_length <= sequence_length + context_length:
        raise ValueError("Data length is too short for the given sequence and context lengths.")

    losses = []
    ys = []  # document verbosity output
    choices = []  # document details of top choices at each position
    ranks = []  # document ranking of each token based on its previous predictions
    # whether to select a random (fixed-length) section to measure
    if random_state:
        rng = np.random.default_rng(random_state)
        begin_loc = rng.integers(low=0, high=len(data) - sequence_length - context_length - 1)
    else:
        begin_loc = 0
    # a reasonably long context length can improve surprisal estimation
    if context_length:
        if len(data) < sequence_length + context_length:
            if not use_all_tokens:
                raise ValueError(f"Input is too short: only {len(data)} tokens provided."
                                 f"Expected more than {sequence_length + context_length} tokens.")
            else:
                warnings.warn(f'Input is shorter than expected; only {len(data)} tokens are used.', RuntimeWarning)
        total_calculated_tokens = 0
        # Strategy:
        # 1. attempt to move forward by (block_size - context_length) tokens each time
        # 2. handle edge cases where the remaining tokens do not fit into block_size but are longer than context_length
        # 3. ignore the last tokens shorter than context_length tokens
        step = block_size - context_length
        # if use_all_tokens, sequence_length is only a recommended, minimum length
        total_length = len(data) if use_all_tokens else sequence_length
        max_iters = total_length // step if not (total_length % step) else total_length // step + 1
        begin_locs = [begin_loc + i * step for i in range(max_iters)]
        for i, begin_loc in enumerate(begin_locs):
            # fit a whole forward pass
            if total_calculated_tokens + block_size <= total_length:
                x = (data[begin_loc: begin_loc + block_size]).astype(np.int64)
                y = (data[begin_loc + 1: begin_loc + block_size + 1]).astype(np.int64)
                num_covered_tokens = context_length
            # fit a forward pass with at least `context_length` tokens
            # fixme: the logic works but not elegant
            elif total_calculated_tokens + step <= total_length < total_calculated_tokens + block_size:
                num_covered_tokens = total_length - begin_loc
                if use_all_tokens:
                    x = (data[begin_loc: -1]).astype(np.int64)
                    y = (data[begin_loc + 1:]).astype(np.int64)
                else:
                    if len(data) - 1 >= block_size + begin_loc:
                        x = (data[begin_loc: begin_loc + block_size]).astype(np.int64)
                        y = (data[begin_loc + 1:  begin_loc + block_size + 1]).astype(np.int64)
                    else:
                        x = (data[begin_loc: -1]).astype(np.int64)
                        y = (data[begin_loc + 1:]).astype(np.int64)
            else:
                continue
            x = torch.from_numpy(x).view(1, -1)  # 1, <=1024
            y = torch.from_numpy(y).view(1, -1)  # 1, <=1024
            # forward pass
            logits, loss = model.forward_reduction_none(x.to(device), y.to(device))
            loss = loss.cpu()  # 1024
            logits = logits.cpu()  # 1, 1024, 50304
            _loss = loss[num_covered_tokens:]
            # casual lm: x_{:i} -> logits[i,:]-> y_i (i.e., x_{:i+1})
            # fixme: 50304 is only for tiktoken's gpt2 tokenizer
            _logits = logits.view(-1, 50304)[num_covered_tokens:, :]  # 50304 is the vocab_size
            _y = y.cpu().numpy()[0][num_covered_tokens:].tolist()  # only for ranking
            # calculate and document results
            _choices, _ranks = get_top_choices_and_ranks(_logits, _y, top_k)
            choices.extend(_choices)
            ranks.extend(_ranks)
            ys.extend(y.cpu().numpy()[0][num_covered_tokens:].tolist())
            losses.extend(_loss.tolist())
            # logic control
            total_calculated_tokens += step
    # move `block_size` forward every time (only for illustrative purposes, never use in practice)
    else:  # context_length == 0
        if len(data) < sequence_length + 1:
            raise ValueError(f"Input is too short: only {len(data)} tokens provided."
                             f"Expected more than {sequence_length + 1} tokens.")
        # ignore last tokens do not naturally fit in a forward pass
        begin_locs = [begin_loc + i * block_size for i in range(sequence_length // block_size)]

        for begin_loc_tmp in begin_locs:
            x = (data[begin_loc_tmp: begin_loc_tmp + block_size]).astype(np.int64)
            y = (data[begin_loc_tmp + 1: begin_loc_tmp + block_size + 1]).astype(np.int64)
            ys.extend(y.tolist())
            x, y = torch.from_numpy(x).view(1, -1), torch.from_numpy(y).view(1, -1)
            logits, loss = model.forward_reduction_none(x.to(device), y.to(device))
            _logits = logits.cpu().view(1024, -1)  # 1024, 50304
            loss = loss.cpu()
            loss_naive = loss.tolist()  # take loss of all tokens in a forward pass
            losses.extend(loss_naive)
            _y = y.cpu().numpy()[0]
            # calculate and document results
            _choices, _ranks = get_top_choices_and_ranks(_logits, _y, top_k)
            choices.extend(_choices)
            ranks.extend(_ranks)
    return ([loss / np.log(2) for loss in losses],  # 2-based surprisal at each position
            choices,  # topic candidates at each position
            ys,  # token ids for the (sampled) sequence used for surprisal calculation
            ranks,  # ranking of the current token based on its previous prediction
            decode([y for y in ys if y != -1]))  # sampled sequence for surprisal calculation


def decode_ids_for_visualization(ids: List[int]) -> List[str]:
    """
    Decode a list of token ids into their corresponding string representations for visualization.

    Args:
        ids: a list of token ids to be decoded.

    Returns:
        a list of strings corresponding to the decoded token ids.
    """

    enc = tiktoken.get_encoding("gpt2")
    return [enc.decode_single_token_bytes(t_id).decode("utf-8", errors='replace') for t_id in ids]


def colorize_text(words: List[str], cross_entropy: List[float]) -> str:
    """
    Colorize a list of words based on their cross-entropy values using a colormap.

    This function takes in a list of tokens and their corresponding cross-entropy values.
    The cross-entropy values are first scaled to lie between 0 and 1 using the MinMaxScaler.
    Each word is then colorized based on its scaled cross-entropy value using the specified colormap.
    The resulting colorized words are combined into a single string, with each word wrapped in a span
    with the appropriate background color.

    Args:
        words: a list of tokens to be colorized.
        cross_entropy: a list of cross-entropy values corresponding to each word.

    Returns:
        a string with each word from the input list colorized based on its scaled cross-entropy value.
    """
    minmax = MinMaxScaler()
    hot = plt.cm.get_cmap('hot', 256)
    new_hot = hot(np.linspace(0.25, 1, 192))  # use the 3/4 of the higher end of cmap "hot"
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('reversed_hot', new_hot[::-1])
    color_array = minmax.fit_transform(np.array(cross_entropy).reshape(-1, 1))
    template = '<span class="barcode" title="surp: {:.2f}" style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color, ce in zip(words, color_array, cross_entropy):
        color = matplotlib.colors.rgb2hex(cmap(color[0])[:3])
        colored_string += template.format(ce, color, word)
    return "[...]" + colored_string + "[...]"
