import os
import torch
import tiktoken
import numpy as np
from typing import List, Set
from model import GPTConfig, GPT
from datasets import load_dataset


def encode(text: str) -> List[int]:
    """
    Encode a document into its corresponding ids found in `titoken`'s encoding.
    """
    enc = tiktoken.get_encoding("gpt2")

    return enc.encode(text, allowed_special={"<|endoftext|>"})


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


def calculate_type_token_ratio(text: str) -> float:
    """
    Compute type-token ratio (TTR) of a document.
    TTR "refers to the ratio of different unique word stems (types) to the total number of words (tokens)", as per
    Wikipedia.
    """
    ids = encode(text)

    return len(set(ids)) / len(ids)


def ignore_func_word_loss_indices(y: list[int],
                                  sliding_window_length: [int | None]) -> list:
    """
    Find the corresponding indices of function word ids in a sequence.

    Args:
        y: a sequence of token ids
    Example:
        indices = ignore_func_word_loss_indices(encode("Is my cat really cute or not?"), None)
        print(indices)  # [2, 4, 7] -> "cat" "cute" and "?"
    """
    func_ids = encode_function_words_into_ids()

    if sliding_window_length:
        y = y[sliding_window_length:]

    return [idx for idx, token_id in enumerate(y) if token_id not in func_ids]


@torch.no_grad()
def calculate_ppl(text: str,
                  model: GPT,
                  ppl_computing_method: str,
                  ignore_function_words: bool,
                  device: str,
                  sequence_length: int = 2048,
                  block_size: int = 1024,
                  sliding_window_length: [int | None] = 512,
                  random_state: [None | int] = None,
                  compile_model=False) -> np.float64:
    """
    Calculate perplexity of a continuous sequence of tokens extracted from a given text.
    The function finds a random chunk of `sliding_window_length+sequence_length` tokens and returns the perplexity score
     of the last `sequence_length` tokens.

    Uncomment the followed section to have a working example.

    Args:
        text: an English string longer than 2,000 words to get stable perplexity
        model: a nano-GPT style casual language model
        ppl_computing_method:
            `naive`: Approximate the perplexity score without a sliding window.
            `long_history`: Approximate the perplexity score with a sliding window: for each `block_size` chunk of
                text, it only returns the perplexity of the last `block_size - sliding_window_length` tokens. This
                method favors global 'surprise' over local grammatical choice. It requires `sliding_window_length` more
                tokens than `naive`
        ignore_function_words: skip computing loss when target is a function word
        device: device used for computation; should be legal in torch.device(), e.g. "cpu", "cuda", and "cuda:1"
        sequence_length: the desired length of tokens whose perplexity score will be returned
        block_size: max sequence length of `model`
        sliding_window_length (int): number of preceding tokens whose loss will not be returned; ignored when
            `ppl_computing_method` set to `naive`
        random_state: supply a random number generator
        compile_model (bool): if True, compile the PyTorch model (require PyTorch 2.0 installed)

    Returns:
        Perplexity score as a float.
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
    data = np.array(encode(text))  # np.array, token ids of the whole document

    losses = []
    if ppl_computing_method == 'long_history':
        if len(data) < sequence_length + sliding_window_length + 1:
            raise RuntimeError(f"`text` too short ({len(data)})."
                               f"Expect `text` length no short than "
                               f"({sequence_length + sliding_window_length + 1}) in terms of tokens.")
        begin_loc = rng.integers(low=0, high=len(data) - sequence_length - sliding_window_length - 1)
        begin_locs = [begin_loc + sliding_window_length * offset for offset in
                      range(sequence_length // sliding_window_length)]
        X = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in begin_locs])
        Y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in begin_locs])
        # calculate ppl of last `block_size - sliding_window_length` tokens
        # always conditioned on at least preceding `sliding_window_length` tokens
        for k in range(sequence_length // sliding_window_length):  # with default settings, four sequences to compute
            x, y = X[k].view(1, -1), Y[k].view(1, -1)
            _, loss = model.forward_reduction_none(x.to(device), y.to(device))
            loss = loss.cpu()
            loss_well_contextualized = loss[
                                       sliding_window_length:].tolist()  # only take nll of last `sliding_window_length` tokens
            if ignore_function_words:
                ignore_indices = ignore_func_word_loss_indices(y.cpu().tolist()[0], sliding_window_length)
                loss_well_contextualized = [loss for idx, loss in enumerate(loss_well_contextualized) if idx not in ignore_indices]
            losses.extend(loss_well_contextualized)


    elif ppl_computing_method == 'naive':
        if len(data) < sequence_length + 1:
            raise RuntimeError(f"`text` too short ({len(data)})."
                               f"Expect `text` length no short than "
                               f"({sequence_length + 1}) in terms of tokens.")

        begin_loc = rng.integers(low=0, high=len(data) - sequence_length - 1)
        begin_locs = [begin_loc + block_size * offset for offset in range(sequence_length // block_size)]

        X = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in begin_locs])
        Y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in begin_locs])

        for k in range(sequence_length // block_size):  # with default settings, only two sequences to compute
            x, y = X[k].view(1, -1), Y[k].view(1, -1)
            _, loss = model.forward_reduction_none(x.to(device), y.to(device))
            loss = loss.cpu()
            loss_naive = loss.tolist()  # take nll of all tokens

            if ignore_function_words:
                ignore_indices = ignore_func_word_loss_indices(y.cpu().tolist()[0], None)
                loss_naive = [loss for idx, loss in enumerate(loss_naive) if idx not in ignore_indices]
            losses.extend(loss_naive)

    return np.exp2(np.mean(losses))


# if __name__ == '__main__':
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
#     # ignore loss targeting function words
#     ppl = calculate_ppl(text=text,
#                         model=model,
#                         ppl_computing_method='long_history',
#                         ignore_function_words=True,
#                         device=device,
#                         sequence_length=2048,
#                         block_size=1024,
#                         sliding_window_length=512,
#                         random_state=0,
#                         compile_model=True)
#     print(ppl)  # ~4.78
#
#     # calculate ppl of a document conditioned on at least 512 preceding tokens
#     # allow loss targeting function words
#     ppl = calculate_ppl(text=text,
#                         model=model,
#                         ppl_computing_method='long_history',
#                         ignore_function_words=False,
#                         device=device,
#                         sequence_length=2048,
#                         block_size=1024,
#                         sliding_window_length=512,
#                         random_state=0,
#                         compile_model=True)
#     print(ppl)  # ~7.85
#
#     # calculate ppl of a document conditioned on preceding tokens within a block size, no sliding window
#     # ignore loss targeting function words
#     ppl = calculate_ppl(text=text,
#                         model=model,
#                         ppl_computing_method='naive',
#                         ignore_function_words=True,
#                         device=device,
#                         sequence_length=2048,
#                         block_size=1024,
#                         sliding_window_length=None,
#                         random_state=0,
#                         compile_model=True)
#     print(ppl)  # ~4.82
#     # calculate ppl of a document conditioned on preceding tokens within a block size, no sliding window
#     # allow loss targeting function words
#     ppl = calculate_ppl(text=text,
#                         model=model,
#                         ppl_computing_method='naive',
#                         ignore_function_words=False,
#                         device=device,
#                         sequence_length=2048,
#                         block_size=1024,
#                         sliding_window_length=None,
#                         random_state=0,
#                         compile_model=True)
#     print(ppl)  # ~8.74
