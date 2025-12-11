import pandas as pd

from pws_in_context.constants import DATA_PATH, device
from pws_in_context.load_model import load_llm
from pws_in_context.utils import (
    all_substrings,
    build_lattice,
    calculate_surprisal,
    dp_prob,
    get_probs,
    token_string,
)

prob_cache = {}


def main():
    target_sent_matching_df = pd.read_csv(DATA_PATH / "target_sent_combinations.csv")

    model, tokenizer, whitespace_char = load_llm(model_key="Llama")

    surprisal_target_list = []
    surprisal_post_target_list = []

    for context, target, post_target in zip(
        target_sent_matching_df["context"].tolist(),
        target_sent_matching_df["target"].tolist(),
        target_sent_matching_df["post_target"].tolist(),
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

    target_sent_matching_df["target_surprisal"] = surprisal_target_list
    target_sent_matching_df["post_target_surprisal"] = surprisal_post_target_list

    target_sent_matching_df.to_csv(DATA_PATH / "target_sent_surprisals.csv", index=False)


if __name__ == "__main__":
    main()
