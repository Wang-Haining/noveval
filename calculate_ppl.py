import os 
import torch
import tiktoken
import numpy as np
from typing import Callable
from model import GPTConfig, GPT
from datasets import load_dataset


@torch.no_grad()
def calculate_ppl(text: str,
                  model: GPT,
                  tokenizer: Callable,
                  device: str,
                  sequence_length: int = 2048,
                  block_size: int = 1024,
                  sliding_window_length: int = 512,
                  random_state: [None or int] = None,
                  compile_model=False) -> np.float64:
    """
    Calculate perplexity of a continuous sequence of tokens extracted from a given text.
    The function finds a random chunk of `sliding_window_length+sequence_length` tokens and returns the perplexity score
     of the last `sequence_length` tokens.

    Note that the function approximates the perplexity score with a sliding window: for each `block_size` chunk of
    text, it only returns the perplexity of the last `block_size - sliding_window_length` tokens. This method favors
    global innovation over local grammatical choice.

    Uncomment the followed section to have a working example.

    Args:
        text: an English string longer than 2,000 words to get stable perplexity
        model: a nano-GPT style casual language model
        tokenizer: a `tiktoken` tokenizer that encodes a string (`text`) to token ids
        device: device used for computation; should be legal in torch.device(), e.g. "cpu", "cuda", and "cuda:1"
        sequence_length: the desired length of tokens whose perplexity score will be returned
        block_size: max sequence length of `model`
        sliding_window_length (int): leading number of tokens whose loss will not be returned
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

    # prepare data
    data = np.array(tokenizer(text))  # np.array, token ids of the whole document
    if len(data) < sequence_length + sliding_window_length + 1:
        raise RuntimeError(f"`text` too short ({len(data)})."
                           f"Expect `text` length no short than "
                           f"({sequence_length + sliding_window_length + 1}) in terms of tokens.")
    begin_loc = rng.integers(low=0, high=len(data) - sequence_length - sliding_window_length)
    begin_locs = [begin_loc + sliding_window_length * offset for offset in range(sequence_length//sliding_window_length)]
    X = torch.stack([torch.from_numpy((data[i: i+block_size]).astype(np.int64)) for i in begin_locs])
    Y = torch.stack([torch.from_numpy((data[i+1: i+1+block_size]).astype(np.int64)) for i in begin_locs])

    # calculate ppl of last `block_size - sliding_window_length` tokens
    # conditioned on previous `sliding_window_length` tokens
    if compile_model:
        model = torch.compile(model)
    model.to(device)
    model.eval()
    losses = []
    for k in range(sequence_length//sliding_window_length):
        x, y = X[k].view(1, -1), Y[k].view(1, -1)
        _, loss = model.forward_reduction_none(x.to(device), y.to(device))
        loss = loss.cpu()
        loss_w_context = loss[sliding_window_length:].tolist()  # only take nll of last `sliding_window_length` tokens
        losses.extend(loss_w_context)

    return np.exp2(np.mean(losses))


# if __name__ == '__main__':
#
#     # load model
#     device = 'cuda:0'
#     out_dir = 'out'
#     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     gptconf = GPTConfig(**checkpoint['model_args'])
#     model = GPT(gptconf)
#     state_dict = checkpoint['model']
#     model.load_state_dict(state_dict)
#
#     # load tokenizer
#     enc = tiktoken.get_encoding("gpt2")
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#
#     # load corpus
#     wikitext_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     text = " ".join(wikitext_test["text"]).replace('\n', '')
#
#     # calculate ppl
#     ppl = calculate_ppl(text=text,
#                         model=model,
#                         tokenizer=encode,
#                         device="cuda:0",
#                         sequence_length=2048,
#                         block_size=1024,
#                         sliding_window_length=512,
#                         random_state=0,
#                         compile_model=True)
#     print(ppl)  # ~9.54
