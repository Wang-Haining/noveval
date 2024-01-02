"""
This script complements the section testing face validity in the paper 'A Novelty Measure for Scholarly Publications
Aligned with Peer Review'.
This module implements token- and sentence-level checks using surprisal from a Wikipedia-exposed GPT as a measure of
novelty found in scholarly publication.

The script loads a pre-trained GPT model and uses it to calculate the surprisal (unexpectedness) of various sentence
continuations. It specifically examines the surprisal of three tokens that incorporate either common or uncommon
(including nonsensical) phrases within a fixed context related to quantum physics and entanglement theory.

Additionally, the module includes a sentence-level test where multiple paraphrased versions of a given sentence are
evaluated for their average surprisal. This allows for a broader understanding of how different formulations of a
concept impact the model's perception of novelty.
"""

import os
from typing import List

import torch
from scipy.stats import ttest_ind

from model import GPT, GPTConfig
from utils import calculate_surprisal, decode, encode


def batch_fixed_context_check(
    context: str, paraphrases: List[str], context_length: int = 512
) -> List[float]:
    """
    Computes the average surprisal (in bits) for each paraphrase when appended to a given context.

    Args:
        context: the fixed context string to prepend to each paraphrase.
        paraphrases: a list of paraphrase strings to be evaluated.
        context_length: number of preceding tokens whose loss will not be returned.

    Returns:
        a list containing the average surprisal score for each paraphrase.

    Raises:
        RuntimeError: if there's a misalignment between the decoded ids and the paraphrase text.
    """
    context_len = len(encode(context))
    paraphrases_len = [len(encode(p)) for p in paraphrases]

    avg_surps = []
    for i, p in enumerate(paraphrases):
        _surps, _, _ids, _, _ = calculate_surprisal(
            text=context + p * 100,  # a hack: make inputs long enough
            model=model,
            context_length=512,
            sequence_length=1024,
            use_all_tokens=False,
            device=device,
            compile_model=True,
        )
        sent_surp = _surps[
            context_len
            - context_length : context_len
            + paraphrases_len[i]
            - context_length
        ]
        avg_surps.append(sum(sent_surp) / len(sent_surp))
        # ensure alignment
        if (
            decode(
                _ids[
                    context_len
                    - context_length : context_len
                    + paraphrases_len[i]
                    - context_length
                ]
            )
            != paraphrases[i]
        ):
            raise RuntimeError("Misalignment found.")

    return avg_surps


if __name__ == "__main__":
    # load model
    device = "cpu"
    out_dir = "out_wikipedia_en"
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    # fmt: off
    # context for all
    # adopted from https://www.nature.com/articles/nphys2904 with inline references stripped
    context = '''Quantum physics started with Max Planck's 'act of desperation', in which he assumed that energy is quantized in order to explain the intensity profile of the black-body radiation. Some twenty-five years later, Werner Heisenberg, Max Born, Pascual Jordan, Erwin Schrödinger and Paul Dirac wrote down the complete laws of quantum theory. A pertinent question then immediately came up — and was subsequently hotly debated by the founding fathers of quantum physics: what features of quantum theory make it different from classical mechanics? Is it Planck's quantization, Bohr's complementarity, Heisenberg's uncertainty principle or the superposition principle?

    Schrödinger felt that the answer was none of the above. In some sense, each of these features can also be either present or mimicked within classical physics: energy can be coarse-grained classically — by brute force if nothing else; waves can be superposed; and complementarity and uncertainty can be found in the trade-off between the knowledge of the wavelength and position of the wave. But the one effect Schrödinger thought had no classical counterpart whatsoever — the characteristic trait of quantum physics — is entanglement.

    The reason entanglement is so counterintuitive and presents a radical departure from classical physics can be nicely explained in terms of modern quantum information theory mixed with some of Schrödinger's jargon. The states of quantum systems are described by what Schrödinger called 'catalogues of information' (psi-wavefunctions). These catalogues contain the probabilities for all possible outcomes of the measurements we can make on the system. Schrödinger thought it odd that when we have two entangled physical systems, their joint catalogue of information can be better specified than the catalogue of each individual system. In other words, the whole can be less uncertain than either of its parts!

    This is impossible, classically speaking. Imagine that someone asks you to predict the toss of a single (fair) coin. Most likely you would not bet too much on it because the outcome is completely uncertain. But consider that tossing two coins becomes less uncertain. Indeed, quantum mechanically, the state of two coins could be completely known, whereas the state of each of the coins is still maximally uncertain.

    In quantum information theory, this leads to negative conditional entropies. When it comes to quantum coins, as we know the outcome, two predictable tosses have zero entropy. However, if we only toss one coin, the outcome is completely uncertain and therefore has one unit of entropy. If we were to quantify the entropy of the second toss, given that the first has been conducted, we would come up with one negative bit — that is, the entropy of two tosses minus the entropy of one toss: 0 − 1 = −1 bit.

    It is precisely because of such peculiarities that the pioneers of quantum physics considered entanglement weird and counterintuitive. However, after around twenty years of intense research in this area, we are now accustomed to entanglement and, moreover, as we learn more about it we discover that entanglement emerges in unexpected places.

    Negative entropies have a physical meaning in thermodynamics. My colleagues and I have shown that negative entropy refers to the situation where we can erase the state of the system, but at the same time obtain some useful work from it. In classical physics we need to invest work in order to erase information — a process known as Landauer's erasure, but quantum mechanically we can have it both ways. This is possible because the system erasing the information could be entangled with the system that is having its information erased. In that case, the total state could have zero entropy, so it can be reset without doing work. Moreover, the eraser now also results in a zero-entropy state and so it can be used to obtain one unit of work.

    Furthermore, we realized that entanglement can exist in many-body systems (with arbitrarily large numbers of particles) as well as at finite temperature. Entanglement can be witnessed using macroscopic observables, such as the heat capacity. In fact, entanglement also serves as an order parameter characterizing quantum phase transitions, and there is growing evidence that quantum topological phase transitions can only be understood in terms of entanglement. A quantum phase transition is a macroscopic change driven by a variation in the ground state of a many-body system at zero temperature. But, in contrast to an ordinary phase, no local order parameter can distinguish between the ordered and the disordered topological phases. For instance, because the change from non-magnetic to magnetic behaviour constitutes an ordinary phase transition, we can check whether an ordinary phase is magnetic by measuring the state of just one spin. However, a topological phase transition cannot be characterized by a local parameter — it requires an understanding of the global entanglement of the whole state.

    This is good news for stable encoding of quantum information. The idea is to use topological phases as quantum memories. This is precisely because topological states are gapped (that is, the energy gap between the ground and excited states is finite) and no local noise can kick the topological state out of the protected subspace. The ground states are also degenerate, meaning that there are different states with the same level of robustness that can be used to encode information.'''
    # fmt: on

    # --------------------
    # Token level test
    # tokens starting immediate after the interested ones (e.g., low and room)
    # start from the first appearance of 'temperature' will not be considered
    # multiply the string by 100 simply make long enough for a forward pass of the model

    # fmt: off
    normal_continuation = """ Building upon this research background, we propose a method that uses photonic crystals at low temperature to observe quantum entanglement.""" * 100

    novel_continuation = """ Building upon this research background, we propose a method that uses photonic crystals at room temperature to observe quantum entanglement.""" * 100

    nonsense_continuation1 = """ Building upon this research background, we propose a method that uses photonic crystals at cat temperature to observe quantum entanglement.""" * 100
    # fmt: on

    # calculate ppl of a document conditioned on at least 512 preceding tokens
    novel = context + novel_continuation
    normal = context + normal_continuation
    nonsense1 = context + nonsense_continuation1

    novel_surps, novel_tops, novel_ids, novel_ranks, _ = calculate_surprisal(
        text=novel,
        model=model,
        context_length=512,
        sequence_length=1024,
        use_all_tokens=False,
        device=device,
        compile_model=True,
    )
    (
        normal_surps,
        normal_tops,
        normal_ids,
        normal_ranks,
        _,
    ) = calculate_surprisal(
        text=normal,
        model=model,
        context_length=512,
        sequence_length=1024,
        use_all_tokens=False,
        device=device,
        compile_model=True,
    )
    (
        nonsense1_surps,
        nonsense1_tops,
        nonsense1_ids,
        nonsense1_ranks,
        _,
    ) = calculate_surprisal(
        text=nonsense1,
        model=model,
        context_length=512,
        sequence_length=1024,
        use_all_tokens=False,
        device=device,
        compile_model=True,
    )

    # the three statements share the same history
    # i.e., 1,184 preceding tokens (in `context`) + 16 tokens in `*continuations` before the interested words
    # so tokens before the interested ones should have the same surprisal scores
    # i.e., tokens before 1 (warm-up token) + 1128 (context length) + 16 (same tokens in continuations) - 512 (history)
    # = 633
    # print(decode(novel_ids[:632]))  # see the shared history across examples
    assert (
        decode(novel_ids[:632])
        == decode(normal_ids[:632])
        == decode(nonsense1_ids[:632])
    )
    assert novel_surps[:632] == normal_surps[:632] == nonsense1_surps[:632]

    # surp('low') < surp('room') < surp('cat')
    assert normal_surps[632] < novel_surps[632] < nonsense1_surps[632]
    # check surprisal scores for 'low', 'room', and 'cat'
    print(f"""Surprisal of token {decode([normal_ids[632]]).strip()} is {normal_surps[632]}.""")  # 2.583745754828677
    print(f"""Surprisal of token {decode([novel_ids[632]]).strip()}is {novel_surps[632]}.""")  # 4.422831740888419
    print(f"""Surprisal of token {decode([nonsense1_ids[632]]).strip()} is {nonsense1_surps[632]}.""")  # 17.5066341068615

    # --------------------
    # Sentence level test
    # 20 paraphrases for each sentence using ChatGPT
    # with prompt "Please paraphrase the sentence 20 times: "
    # fmt: off
    normal_paraphrases = [
        " Based on the foundation of prior research, we suggest a technique employing photonic crystals at reduced temperatures for the observation of quantum entanglement.",
        " Advancing from the existing research, our approach involves using photonic crystals in cold environments to study quantum entanglement.",
        " Expanding on previous studies, we introduce a method that utilizes photonic crystals in low-temperature settings to examine quantum entanglement.",
        " Drawing from earlier research, we present a strategy that applies photonic crystals under cool conditions to detect quantum entanglement.",
        " Leveraging the background of existing research, our proposed method involves photonic crystals at low temperatures to explore quantum entanglement.",
        " From the groundwork of prior studies, we put forward a technique using photonic crystals in chilly temperatures to observe quantum entanglement.",
        " With the foundation of previous research, we propose a new method that employs photonic crystals at a low temperature to study quantum entanglement.",
        " Progressing from existing studies, our method incorporates the use of photonic crystals in colder temperatures for quantum entanglement observation.",
        " Building on the base of established research, we advocate a method that involves photonic crystals in frigid environments to scrutinize quantum entanglement.",
        " Developing from earlier research, we introduce a process that leverages photonic crystals at reduced temperatures to detect quantum entanglement.",
        " Extending upon previous research findings, we suggest a technique that applies photonic crystals in low-temperature conditions to analyze quantum entanglement.",
        " Taking inspiration from past research, our proposed method uses photonic crystals in cool settings to observe quantum entanglement.",
        " Progressing from the research foundation, we offer a method utilizing photonic crystals at a lower temperature to examine quantum entanglement.",
        " Advancing previous research, we devise a method that employs photonic crystals in colder conditions to study quantum entanglement.",
        " Following the path of prior research, we propose a technique that involves photonic crystals under low temperatures for observing quantum entanglement.",
        " Enhancing the existing research base, our method uses photonic crystals in chilly environments to analyze quantum entanglement.",
        " Evolving from earlier studies, we propose a technique utilizing photonic crystals at reduced temperatures for quantum entanglement examination.",
        " On the shoulders of previous research, we propose a method involving photonic crystals in cold settings to explore quantum entanglement.",
        " Furthering past research, we suggest a method that makes use of photonic crystals at low temperatures for the observation of quantum entanglement.",
        " Capitalizing on existing research, our approach includes using photonic crystals in cool temperatures to scrutinize quantum entanglement.",
    ]

    novel_paraphrases = [
        "Based on the foundation of this research, we suggest a technique that employs photonic crystals at ambient temperature for observing quantum entanglement.",
        "Expanding from the existing research, our approach involves using photonic crystals at standard room temperature to study quantum entanglement.",
        "Advancing the current research, we introduce a method utilizing photonic crystals in normal temperature conditions to examine quantum entanglement.",
        "Drawing on prior research, we present a strategy that applies photonic crystals at average room temperatures to detect quantum entanglement.",
        "Leveraging this research background, we propose a method involving photonic crystals at comfortable room temperatures to explore quantum entanglement.",
        "From the groundwork of earlier studies, we offer a technique using photonic crystals in typical room temperature settings to observe quantum entanglement.",
        "Utilizing the foundation of past research, we propose a new method employing photonic crystals at standard living temperatures to study quantum entanglement.",
        "Progressing from existing studies, our method incorporates photonic crystals in ordinary room temperatures for quantum entanglement observation.",
        "Building on established research, we advocate a method involving photonic crystals in everyday temperature environments to scrutinize quantum entanglement.",
        "Developing from earlier research, we introduce a process that leverages photonic crystals at normal room temperatures to detect quantum entanglement.",
        "Extending upon prior research findings, we suggest a technique applying photonic crystals in average indoor temperatures to analyze quantum entanglement.",
        "Inspired by past research, we propose a method using photonic crystals in standard ambient temperatures to observe quantum entanglement.",
        "Progressing from the research base, we offer a method that utilizes photonic crystals at comfortable indoor temperatures to examine quantum entanglement.",
        "Advancing previous research, we devise a method that employs photonic crystals in regular room conditions to study quantum entanglement.",
        "Following the trajectory of prior research, we propose a technique involving photonic crystals under room temperatures for observing quantum entanglement.",
        "Enhancing existing research, our method uses photonic crystals in typical indoor environments to analyze quantum entanglement.",
        "Evolving from earlier studies, we propose a technique that utilizes photonic crystals at normal indoor temperatures for quantum entanglement examination.",
        "Standing on the shoulders of previous research, we propose a method involving photonic crystals in everyday room settings to explore quantum entanglement.",
        "Furthering past research, we suggest a method making use of photonic crystals at average temperatures for the observation of quantum entanglement.",
        "Capitalizing on existing research, our approach includes using photonic crystals at standard room temperatures to scrutinize quantum entanglement.",
    ]
    # fmt: on

    normal_avg_surps = batch_fixed_context_check(context, normal_paraphrases)
    novel_avg_surps = batch_fixed_context_check(context, novel_paraphrases)

    # # boxplot
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.boxplot([normal_avg_surps, novel_avg_surps], labels=["Normal", "Novel"])
    # plt.title("Boxplot of Average Surprisal Values")
    # plt.ylabel("Average Surprisal")
    # plt.grid(True)
    # plt.show()

    # one-tailed Welch’s unequal variances t-test
    print(
        ttest_ind(
            novel_avg_surps,
            normal_avg_surps,
            equal_var=False,
            alternative="greater",
        )
    )
    # TtestResult(statistic=2.9237765758084704, pvalue=0.002953316723794009, df=36.52733499644924)
