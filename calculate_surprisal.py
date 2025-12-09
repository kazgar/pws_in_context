import math

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data"

prob_cache = {}

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

model.eval()

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)

model.to(device)


def all_substrings(S: str):
    subs = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n + 1):
            subs.add(S[i:j])
    return subs


def token_string(tok: str, whitespace_char: str = "Ġ"):
    return tok.replace(whitespace_char, " ")


def build_lattice(S: str, token_strings: list[str]):
    n = len(S)
    lattice = [[] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for tok in token_strings:
            j = i - len(tok)
            if j >= 0 and S[j:i] == tok:
                lattice[i].append((j, tok))
    return lattice


def get_probs(prefix: str, pruned_tokens: list[str], tok_to_id: dict):
    global prob_cache

    if prefix in prob_cache:
        return prob_cache[prefix]

    ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1]

    probs = torch.softmax(logits, dim=-1)

    result = {}
    for tok in pruned_tokens:
        tid = tok_to_id[tok]
        result[tok] = float(probs[tid].cpu())

    prob_cache[prefix] = result
    return result


def dp_prob(S: str, lattice: dict[str, list[str]], pruned_tokens: list[str], tok_to_id: dict):
    n = len(S)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0

    for i in range(1, n + 1):
        for j, tok in lattice[i]:
            prefix = S[:j]
            p = get_probs(prefix, pruned_tokens, tok_to_id)[tok]
            contrib = dp[j] * p
            dp[i] += contrib

    return dp[n]


def calculate_surprisal(context: str, target: str, tokenizer: AutoTokenizer):
    global prob_cache

    prob_cache = {}

    sentence = context + " " + target
    substring_set = all_substrings(sentence)

    id2token = [tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)]

    token_strings = [token_string(tok) for tok in id2token]

    pruned_tokens = []
    pruned_token_ids = []

    for tid, t in enumerate(token_strings):
        if t in substring_set:
            pruned_tokens.append(t)
            pruned_token_ids.append(tid)

    tok_to_id = dict(zip(pruned_tokens, pruned_token_ids))

    lattice_context = build_lattice(context, pruned_tokens)
    lattice_sentence = build_lattice(sentence, pruned_tokens)

    P_context = dp_prob(context, lattice_context, pruned_tokens, tok_to_id)
    P_sentence = dp_prob(sentence, lattice_sentence, pruned_tokens, tok_to_id)

    try:
        P_cond = P_sentence / P_context
    	surprisal = -math.log(P_cond)
    except:
	surprisal = None

    return surprisal


def main():
    target_sent_matching_df = pd.read_csv(DATA_PATH / "target_sent_matching.csv")

    surprisals = []

    for context, target in zip(
        target_sent_matching_df["context"].tolist(), target_sent_matching_df["target"].tolist()
    ):
        surprisals.append(calculate_surprisal(context, target, tokenizer))

    target_sent_matching_df["surprisal"] = surprisals

    target_sent_matching_df.to_csv(DATA_PATH / "target_sent_surprisals.csv", index=False)


if __name__ == "__main__":
    main()
