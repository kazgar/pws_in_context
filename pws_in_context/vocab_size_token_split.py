import os

import pandas as pd

from pws_in_context.constants import DATA_PATH, device
from pws_in_context.load_model import load_llm, models


def main():
    target_sent_combinations_df = pd.read_csv(DATA_PATH / "target_sent_combinations.csv")
    unique_targets = target_sent_combinations_df["target"].unique().tolist()

    vocab_size_dict = {"model": [], "tokenizer_vocab_size": [], "model_vocab_size": []}

    token_split_dict = {
        "target": [],
        "number_of_tokens": [],
        "number_of_tokens_w_whitespace": [],
        "tokens": [],
        "tokens_w_whitespace": [],
        "model": [],
    }

    for model_name in models:
        model, tokenizer, _ = load_llm(model_name)
        tokenizer_vocab_size = tokenizer.vocab_size
        model_vocab_size = model.config.vocab_size

        del model

        vocab_size_dict["model"].append(model_name)
        vocab_size_dict["tokenizer_vocab_size"].append(tokenizer_vocab_size)
        vocab_size_dict["model_vocab_size"].append(model_vocab_size)

        for target in unique_targets:
            tokens = tokenizer(target, add_special_tokens=False).tokens()
            tokens_with_whitespace = tokenizer(f" {target}", add_special_tokens=False).tokens()

            token_split_dict["target"].append(target)
            token_split_dict["number_of_tokens"].append(len(tokens))
            token_split_dict["number_of_tokens_w_whitespace"].append(len(tokens_with_whitespace))
            token_split_dict["tokens"].append(tokens)
            token_split_dict["tokens_w_whitespace"].append(tokens_with_whitespace)
            token_split_dict["model"].append(model_name)

    vocab_size_df = pd.DataFrame.from_dict(vocab_size_dict)

    if not os.path.exists(DATA_PATH / "vocab_size_and_token_splits"):
        os.makedirs(DATA_PATH / "vocab_size_and_token_splits", exist_ok=True)

    vocab_size_df.to_csv(DATA_PATH / "vocab_size_and_token_splits" / "vocab_sizes.csv", index=False)

    token_split_df = pd.DataFrame.from_dict(token_split_dict)
    token_split_df.to_csv(
        DATA_PATH / "vocab_size_and_token_splits" / "token_splits.csv", index=False
    )


if __name__ == "__main__":
    main()
