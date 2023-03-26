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
                  stride: int = 512,
                  random_state: [None | int] = None,
                  compile=False):
    """
    Calculate perplexity of a continuous sequence of tokens extracted from a given, perhaps long, document.

    TODO: only concern perplexity of tokens properly contextualized (i.e., tokens with at least 512 tokens in the context)
    Args:
        text: an English string, recommended to be longer than 2,000 words to get stable perplexity
        model: a nano-GPT style casual language model
        tokenizer: a tiktoken tokenizer encodes a string (i.e., `text`) to token ids
        device: str, committed device(s) used for calculation; should be legal in torch.device(), e.g., "cuda", "cpu",
            "cuda:1"
        block_size: int, max sequence length of `model`
        stride: int, TODO
        random_state: supply a random number generator
        compile: support pytorch 2.0 compile()

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
    if len(data) < sequence_length + 1:
        raise RuntimeError(f"Number of encoded tokens ({len(data)}) is less than sequence length ({sequence_length}).")  # todo: < or <=
    begin_loc = rng.integers(len(data) - sequence_length)
    begin_locs = [begin_loc + stride * offset for offset in range(sequence_length//stride)]  # todo: test
    X = torch.stack([torch.from_numpy((data[i: i+block_size]).astype(np.int64)) for i in begin_locs])
    Y = torch.stack([torch.from_numpy((data[i+1: i+1+block_size]).astype(np.int64)) for i in begin_locs])

    # calculate ppl of x[512:]
    if compile:
        model = torch.compile(model)
    model.eval()
    losses = []
    for k in range(sequence_length//stride):
        x, y = X[k].view(1, -1), Y[k].view(1, -1)
        _, loss = model.forward_reduction_none(x.to(device), y.to(device))
        loss = loss.cpu()
        loss_w_context = loss[stride:].tolist()  # only take the nll of last `stride` tokens
        losses.extend(loss_w_context)

    return np.exp2(np.mean(losses))

#
# if __name__ == '__main__':
#
#     # load from a model saved in a specific directory
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
#     # decode = lambda l: enc.decode(l)
#
#     wikitext_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     text = " ".join(wikitext_test["text"]).replace('\n', '')
#
#     ppl = calculate_ppl(text=text,
#                         model=model,
#                         tokenizer=encode,
#                         device="cuda:0",
#                         sequence_length=2048,
#                         block_size=1024,
#                         stride=512,
#                         random_state=0,
#                         compile=True)


