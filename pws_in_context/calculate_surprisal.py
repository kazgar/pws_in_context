import math

import pandas as pd
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerBase

from pws_in_context.constants import DATA_PATH, device
from pws_in_context.load_model import load_llm

prob_cache = {}


def all_substrings(s: str) -> set[str]:
    """Return all substrings of `s`"""
    subs = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            subs.add(s[i:j])
    return subs


def token_string(token: str, whitespace_char: str = "Ġ") -> str:
    """Convert tokenizer token into its string form by replacing whitespace markers."""
    return token.replace(whitespace_char, " ")


def build_lattice(s: str, token_strings: list[str]) -> list[list[tuple[int, str]]]:
    """Build lattice[i] = list of (j, token) such that s[j:i] == token."""
    n = len(s)
    lattice = [[] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for tok in token_strings:
            j = i - len(tok)
            if j >= 0 and s[j:i] == tok:
                lattice[i].append((j, tok))
    return lattice


def get_probs(
    prefix: str,
    pruned_tokens: list[str],
    tok_to_id: dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    prob_cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Return P(next_token | prefix) for each token in pruned_tokens.

    Cached to avoid recomputation.
    """
    if prefix in prob_cache:
        return prob_cache[prefix]

    ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(ids).logits[0, -1]

    probs = torch.softmax(logits, dim=-1)

    result = {}
    for tok in pruned_tokens:
        tid = tok_to_id[tok]
        result[tok] = float(probs[tid].cpu())

    prob_cache[prefix] = result
    return result


def dp_prob(
    s: str,
    lattice: list[list[tuple[int, str]]],
    pruned_tokens: list[str],
    tok_to_id: dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    prob_cache: dict[str, dict[str, float]],
) -> float:
    """Compute probability of s via dynamic programming."""
    n = len(s)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0

    for i in range(1, n + 1):
        for j, tok in lattice[i]:
            prefix = s[:j]
            p = get_probs(prefix, pruned_tokens, tok_to_id, tokenizer, model, device, prob_cache)[
                tok
            ]
            contrib = dp[j] * p
            dp[i] += contrib

    return dp[n]


def calculate_surprisal(
    context: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    whitespace_char: str,
    post_target: str | None = None,
    calc_post: bool = True,
) -> float | tuple[float, float]:
    """Compute surprisal of target (+ optionally post target) given context."""
    prob_cache: dict[str, dict[str, float]] = {}

    sentence = f"{context} {target}"

    substring_set = all_substrings(sentence)

    id2token = [tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)]
    token_strings = [token_string(tok, whitespace_char) for tok in id2token]

    pruned_tokens = []
    pruned_token_ids = []

    for tid, tok_str in enumerate(token_strings):
        if tok_str in substring_set:
            pruned_tokens.append(tok_str)
            pruned_token_ids.append(tid)

    tok_to_id = dict(zip(pruned_tokens, pruned_token_ids))

    lattice_context = build_lattice(context, pruned_tokens)
    lattice_sentence = build_lattice(sentence, pruned_tokens)

    P_context = dp_prob(
        context, lattice_context, pruned_tokens, tok_to_id, tokenizer, model, device, prob_cache
    )

    P_sentence = dp_prob(
        sentence, lattice_sentence, pruned_tokens, tok_to_id, tokenizer, model, device, prob_cache
    )

    compute_post = calc_post and (post_target is not None)

    if compute_post:
        sentence_post = f"{sentence} {post_target}"
        substring_set_post = all_substrings(sentence_post)

        for tid, tok_str in enumerate(token_strings):
            if tok_str in substring_set_post and tok_str not in pruned_tokens:
                pruned_tokens.append(tok_str)
                pruned_token_ids.append(tid)

        tok_to_id_post = dict(zip(pruned_tokens, pruned_token_ids))

        lattice_sentence_post = build_lattice(sentence_post, pruned_tokens)

        P_sentence_post = dp_prob(
            sentence_post,
            lattice_sentence_post,
            pruned_tokens,
            tok_to_id_post,
            tokenizer,
            model,
            device,
            prob_cache,
        )

    surprisal_target = None
    surprisal_post_target = None

    try:
        P_target = P_sentence / P_context
        surprisal_target = -math.log(P_target)

        if compute_post:
            P_post_target = P_sentence_post / P_sentence
            surprisal_post_target = -math.log(P_post_target)
    except Exception as e:
        print(f"Error computing surprisal: {e}")

    if compute_post:
        return surprisal_target, surprisal_post_target
    return surprisal_target


def main():
    target_sent_matching_df = pd.read_csv(DATA_PATH / "target_sent_combinations.csv")

    model, tokenizer, whitespace_char = load_llm(model_key="Llama")

    surprisal_target_list = []
    surprisal_post_target_list = []

    for context, target, post_target in zip(
        target_sent_matching_df["context"].tolist()[:10],
        target_sent_matching_df["target"].tolist()[:10],
        target_sent_matching_df["post_target"].tolist()[:10],
    ):
        target_surprisal, post_target_surprisal = calculate_surprisal(
            context=context,
            target=target,
            tokenizer=tokenizer,
            model=model,
            whitespace_char=whitespace_char,
            post_target=post_target,
            calc_post=True,
            device=device,
        )
        surprisal_target_list.append(target_surprisal)
        surprisal_post_target_list.append(post_target_surprisal)

    print(surprisal_target_list)
    print(surprisal_post_target_list)
    # target_sent_matching_df["target_surprisal"] = surprisal_target_list
    # target_sent_matching_df["post_target_surprisal"] = surprisal_post_target_list
    #
    # target_sent_matching_df.to_csv(DATA_PATH / "target_sent_surprisals.csv", index=False)


if __name__ == "__main__":
    main()
