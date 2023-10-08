import os
import re
import math
import json
import torch
import tiktoken
import matplotlib
import numpy as np
from model import GPT
from itertools import chain
import matplotlib.pyplot as plt
from typing import List, Set, Tuple
from sklearn.preprocessing import MinMaxScaler


def get_paper_and_score(corpus_path: str = "./PeerRead/data/acl_2017/", preserve_ordinal=True):
    """
    Read in papers and ratings on originality from ACL subset of PeerRead.
    """
    aspects = ['ORIGINALITY']

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

    text_dict = [{int(paper['name'].split('.pdf')[0]): clean_up_artifacts("\n\n".join(
        [d['text'] for d in paper['metadata']['sections']]))} for paper in papers]
    title_dict = [{int(paper['name'].split('.pdf')[0]): paper['metadata']['title']} for paper in papers]
    text_sorted_list = [list(d.values())[0] for d in sorted(text_dict, key=lambda x: list(x.keys())[0])]
    text_check_list = [list(d.keys())[0] for d in sorted(text_dict, key=lambda x: list(x.keys())[0])]
    title_sorted_list = [list(d.values())[0] for d in sorted(title_dict, key=lambda x: list(x.keys())[0])]

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
    review_scores.update({'title': title_sorted_list})

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


def clean_up_artifacts(text_from_parsed_pdf):
    """
    Clean up obvious artifacts seen in the parsed pdfs.
    """
    text = re.sub(r"\n\d+", "", text_from_parsed_pdf)
    text = text.replace("1 000\n\n", '')
    text = text.replace("1 000\n", '')
    text = re.sub(r'\"{10,}', "", text)

    return text


def encode_function_words_into_ids() -> Set[int]:
    """
    Covert 512 function words into ids found in `titoken`'s encoding.
    Note that, BPE encodes the same token different given context. See examples below.
        > encode('is')
        [271]
        > encode(' is')
        [318]
    We hack this by manually expanding the function word list with capitalized words and words with preceding space.
    1591 ids are identified.
    """
    koppel512_raw = open('resource/koppel_function_words.txt', 'r').read().split('\n')
    expanded_koppel512 = ([w for w in koppel512_raw] +
                          [" " + w for w in koppel512_raw] +
                          [w.capitalize() for w in koppel512_raw] +
                          [" " + w.capitalize() for w in koppel512_raw])
    func_ids = []
    for func_word in expanded_koppel512:
        func_ids.extend(encode(func_word))

    return set(func_ids)


def ignore_func_word_loss_indices(y: list[int],
                                  minimum_context_length: [int | None]) -> list:
    """
    Find the corresponding indices of function word ids in a sequence.

    Args:
        y: a sequence of token ids
        minimum_context_length: refer to `minimum_context_length` of calculate_perplexity()
    Example:
        indices = ignore_func_word_loss_indices(encode("Is my cat really cute or not?"), None)
        print(indices)  # [2, 4, 7] -> "cat" "cute" and "?"
    """
    func_ids = encode_function_words_into_ids()

    if minimum_context_length:
        y = y[minimum_context_length:]

    return [idx for idx, token_id in enumerate(y) if token_id not in func_ids]


# def calculate_type_token_ratio(text: str,
#                                sampling: bool = True,
#                                sequence_length: int = 2048,
#                                random_state: [None | int] = None) -> float:
#     """
#     Compute type-token ratio (TTR) of a document.
#     TTR "refers to the ratio of different unique word stems (types) to the total number of words (tokens)", as per
#     Wikipedia.
#     """
#     if sampling:
#         # prepare rng
#         if random_state is None:
#             random_state = np.random.randint(np.iinfo(np.int32).max)
#         else:
#             if not 0 <= random_state <= np.iinfo(np.int32).max:
#                 raise ValueError(f"Expect int >= 0 for `random_state`, but got {random_state}.")
#         rng = np.random.default_rng(random_state)
#
#         # prepare data
#         data = np.array(encode(text))  # np.array, token ids of the whole document
#
#         begin_loc = rng.integers(low=0, high=len(data) - sequence_length)
#         x = data[begin_loc: begin_loc + sequence_length]
#     else:
#         x = encode(text)  # decode and encode can be fuzzy, so leave a safe margin
#         if not (sequence_length - 2 <= len(x) <= sequence_length + 2):
#             print(f"{text=}")
#             print(f"{decode(encode(text))}")
#             print(f"{x=}")
#             raise ValueError(f'Without sampling, expect input length of {sequence_length=}, but got {len(x)}.')
#         if not len(x) == sequence_length:
#             print(f"Input ({len(x)=}) and {sequence_length=} not match exactly.")
#     return len(set(x)) / len(x)


@torch.no_grad()
def calculate_perplexity(text: str,
                         model: GPT,
                         computing_method: str,
                         device: str,
                         sequence_length: int = 2048,
                         block_size: int = 1024,
                         minimum_context_length: int = 512,
                         add_initial_eot: bool = True,
                         sampling: bool = True,
                         random_state: [None | int] = None,
                         compile_model: bool = False,
                         verbosity: bool = False) -> [np.float64 |
                                                      Tuple[np.float64, List[np.float64], list, str]]:
    """
    Calculate perplexity of a continuous sequence of tokens extracted from a given text.
    The function finds a random chunk of `minimum_context_length + sequence_length` tokens and returns the perplexity
    score of the last `sequence_length` tokens.

    Uncomment the commented-out to have examples.
    Implementation follows code and advices from
        - https://huggingface.co/docs/transformers/perplexity
        - https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/ac4135177bfee71b1efd7bd3aff62e456e30aef9/\
            perplexity.py
        - https://thegradient.pub/understanding-evaluation-metrics-for-language-models/


    Args:
        text: an English string longer than 2,000 words to get stable perplexity
        model: a nano-GPT style casual language model
        computing_method:
            `naive`: approximate the perplexity score by moving `block_size` tokens forward every time
            `long_history`: approximate the perplexity score with a constraint that every nll is calculated with at
                least `minimum_context_length`: for each `block_size` chunk of text, it only returns the perplexity of
                 the last `block_size - minimum_context_length` tokens. This method favors global 'surprise' over local
                 grammatical choice. It requires `minimum_context_length` more tokens than `naive`.
        device: device used for computation; should be legal in torch.device(), e.g. "cpu", "cuda", and "cuda:1"
        sequence_length: the desired length of tokens whose perplexity score will be returned
        block_size: max sequence length of `model`
        minimum_context_length: number of preceding tokens whose loss will not be returned; ignored when
            `computing_method` set to `naive`
        add_initial_eot: whether include the eot token `<|endoftext|>` in the very beginning.
        sampling: whether subsample `sequence_length` tokens from a longer document; if set false and
            `computing_method` is `long_history`, return perplexity after the first `minimum_context_length` tokens;
             if set false and `computing_method` is `naive`, return perplexity from the start of the document
        random_state: supply a random number generator
        compile_model: if True, compile the PyTorch model (require PyTorch 2.0 installed)
        verbosity: set true to return intermediate variables (x, y, and the corresponding raw losses); otherwise only
            perplexity returned; useful for interpretation.

    Returns:
        If `verbosity`, perplexity score as a float; otherwise return a tuple of perplexity, raw 2-based cross entropy,
         and string used for computation
    """
    # prepare rng
    if random_state is None:
        random_state = np.random.randint(np.iinfo(np.int32).max)
    else:
        if not 0 <= random_state <= np.iinfo(np.int32).max:
            raise ValueError(f"Expect int >= 0 for `random_state`, but got {random_state}.")
    rng = np.random.default_rng(random_state)

    # prepare model
    if compile_model:
        model = torch.compile(model)
    model.to(device)
    model.eval()

    # prepare data
    if add_initial_eot:
        # token ids of the whole document starting with eot
        data = np.array(encode("<|endoftext|>" + text))  # note "<|endoftext|>" id 50256
    else:
        data = np.array(encode(text))  # token ids of the whole document

    # sanity check
    data_length = len(data)
    if data_length <= sequence_length + minimum_context_length:
        raise ValueError("Data length is too short for the given sequence and context lengths.")

    losses = []
    xs = []  # document for verbosity output
    if sampling:
        begin_loc = rng.integers(low=0, high=len(data) - sequence_length - minimum_context_length - 1)
    else:
        begin_loc = 0
    # move (block_size - minimum_context_length) forward every time, if possible
    if computing_method == 'long_history' and minimum_context_length:
        if len(data) < sequence_length + minimum_context_length + 1:
            raise ValueError(f"`text` too short ({len(data)})."
                             f"Expect `text` length no short than "
                             f"({sequence_length + minimum_context_length + 1}) in terms of tokens.")
        total_calculated_tokens = 0
        remaining_tokens = 0
        max_iters = sequence_length // (block_size - minimum_context_length) if not (sequence_length % (
                block_size - minimum_context_length)) else sequence_length // (block_size - minimum_context_length) + 1
        begin_locs = [begin_loc + i * (block_size - minimum_context_length) for i in range(max_iters)]
        for begin_loc_tmp in begin_locs:
            if total_calculated_tokens + (block_size - minimum_context_length) <= sequence_length:
                x = (data[begin_loc_tmp: begin_loc_tmp + block_size]).astype(np.int64)
                y = (data[begin_loc_tmp + 1: begin_loc_tmp + block_size + 1]).astype(np.int64)
                # xs.extend(x)
                x, y = torch.from_numpy(x), torch.from_numpy(y)
            else:
                remaining_tokens = sequence_length - total_calculated_tokens
                num_ignore_masks = block_size - remaining_tokens
                x = (data[begin_loc_tmp: begin_loc_tmp + remaining_tokens]).astype(np.int64)
                y = (data[begin_loc_tmp + 1: begin_loc_tmp + remaining_tokens + 1]).astype(np.int64)
                # xs.extend(x)
                x = torch.cat((torch.from_numpy(x), torch.full((num_ignore_masks,), 1)),
                              dim=0).long()  # add dummy input 1, will be ignored anyway
                y = torch.cat((torch.from_numpy(y), torch.full((num_ignore_masks,), -1)),
                              dim=0).long()  # ignore_index is -1
            x, y = x.view(1, -1), y.view(1, -1)
            _, loss = model.forward_reduction_none(x.to(device), y.to(device))
            loss = loss.cpu()
            # take nll of tokens after the first `minimum_context_length`
            reserve_idx = np.arange(minimum_context_length, block_size
                                    ) if not remaining_tokens else np.arange(
                minimum_context_length, minimum_context_length + remaining_tokens)
            loss_long_history = loss[reserve_idx]
            xs.extend(x.cpu().numpy()[0][reserve_idx].tolist())
            losses.extend(loss_long_history.tolist())
            total_calculated_tokens += (block_size - minimum_context_length)

    # move `block_size` forward every time
    if computing_method == 'naive':
        if len(data) < sequence_length + 1:
            raise RuntimeError(f"`text` too short ({len(data)})."
                               f"Expect `text` length no short than "
                               f"({sequence_length + 1}) in terms of tokens.")

        begin_locs = [begin_loc + i * block_size for i in range(sequence_length // block_size)]

        for begin_loc_tmp in begin_locs:  # with default settings, only two sequences to compute
            x = (data[begin_loc_tmp: begin_loc_tmp + block_size]).astype(np.int64)
            y = (data[begin_loc_tmp + 1: begin_loc_tmp + block_size + 1]).astype(np.int64)
            xs.extend(x.tolist())
            x, y = torch.from_numpy(x).view(1, -1), torch.from_numpy(y).view(1, -1)
            _, loss = model.forward_reduction_none(x.to(device), y.to(device))
            loss = loss.cpu()
            loss_naive = loss.tolist()  # take nll of all tokens
            losses.extend(loss_naive)

    if not len(xs) == sequence_length:
        raise ValueError(f"Output length ({len(xs)}) does not match {sequence_length}.")
    if not verbosity:
        return np.exp(np.mean(losses))
    if verbosity:
        return (np.exp(np.mean(losses)),
                [loss / np.log(2) for loss in losses],  # report 2-based cross-entropy
                xs,
                decode(xs))


def decode_ids_for_visualization(ids: List[int]) -> List[str]:
    """
    Decode a list of token ids into their corresponding string representations for visualization purposes.

    Args:
        ids: A list of token ids to be decoded.

    Returns:
        A list of strings corresponding to the decoded token ids.
    """

    enc = tiktoken.get_encoding("gpt2")
    return [enc.decode_single_token_bytes(t_id).decode("utf-8", errors='replace') for t_id in ids]


def colorize_text(words: List[str], cross_entropy: List[float]) -> str:
    """
    Colorize a list of words based on their cross-entropy values using a colormap.

    This function takes in a list of words and their corresponding cross-entropy values.
    The cross-entropy values are first scaled to lie between 0 and 1 using the MinMaxScaler.
    Each word is then colorized based on its scaled cross-entropy value using the specified colormap.
    The resulting colorized words are combined into a single string, with each word wrapped in a span
    with the appropriate background color.

    Args:
        words: A list of words to be colorized.
        cross_entropy: A list of cross-entropy values corresponding to each word.

    Returns:
        A string with each word from the input list colorized based on its scaled cross-entropy value.
    """
    minmax = MinMaxScaler()
    hot = plt.cm.get_cmap('hot', 256)
    new_hot = hot(np.linspace(0.25, 1, 192))  # use the 3/4 of the higher end of cmap "hot"
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('reversed_hot', new_hot[::-1])
    color_array = minmax.fit_transform(np.array(cross_entropy).reshape(-1, 1))
    template = '<span class="barcode" title="Cross Entropy: {:.2f}" style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color, ce in zip(words, color_array, cross_entropy):
        color = matplotlib.colors.rgb2hex(cmap(color[0])[:3])
        colored_string += template.format(ce, color, word)
    return "[...]" + colored_string + "[...]"


# if __name__ == '__main__':
#     from model import GPTConfig
#     from datasets import load_dataset
#
#     # load model
#     device = 'cuda:0'
#     out_dir = 'out_wikipedia_en'
#     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     gptconf = GPTConfig(**checkpoint['model_args'])
#     model = GPT(gptconf)
#     state_dict = checkpoint['model']
#     model.load_state_dict(state_dict)
#
#     # load corpus
#     wikitext_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     text = " ".join(wikitext_test["text"]).replace('\n', '')
#
#     # calculate ppl of a document conditioned on at least 512 preceding tokens
#     ppl = calculate_perplexity(text=text,
#                                model=model,
#                                computing_method='long_history',
#                                device=device,
#                                sequence_length=2048,
#                                block_size=1024,
#                                minimum_context_length=512,
#                                sampling=True,
#                                random_state=0,
#                                compile_model=True,
#                                verbosity=False)
#     print(ppl)  # ~7.85
#     # calculate ppl of a document conditioned on at least 512 preceding tokens, with verbosity turned on
#     ppl, loss, x, y = calculate_perplexity(text=text,
#                                            model=model,
#                                            computing_method='long_history',
#                                            device=device,
#                                            sequence_length=2048,
#                                            block_size=1024,
#                                            minimum_context_length=512,
#                                            sampling=True,
#                                            random_state=0,
#                                            compile_model=True,
#                                            verbosity=True)
#     print(ppl)  # ~7.85
#
#     # calculate ppl of a document naively by moving 1024 tokens per calculation
#     ppl = calculate_perplexity(text=text,
#                                model=model,
#                                computing_method='naive',
#                                device=device,
#                                sequence_length=2048,
#                                block_size=1024,
#                                sampling=True,
#                                random_state=0,
#                                compile_model=True,
#                                verbosity=False)
#     print(ppl)  # ~7.91
#
#     # calculate ttr of a document by randomly samples a fixed-size of content
#     ttr = calculate_type_token_ratio(text=text,
#                                      sequence_length=2048,
#                                      random_state=0)
#     print(ttr)  # ~0.35
