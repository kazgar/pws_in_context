import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

models = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",  # 1B parameters | 8 languages
    "gpt2": "openai-community/gpt2",  # 0.1B parameters | English
    "bloom-1b1": "bigscience/bloom-1b1",  # 1B parameters | 48 languages
    "bloom-1b7": "bigscience/bloom-1b7",  # 2B parameters | 48 languages
    "opt-125m": "facebook/opt-125m",  # 125M parameters | English
    "opt-1.3b": "facebook/opt-1.3b",  # 1.3B parameters | English
    "falcon-7b": "tiiuae/falcon-7b",  # 7B parameters | English
    "falcon-40b": "tiiuae/falcon-40b",  # 42B parameters | 4 languages
    "pythia-70m": "EleutherAI/pythia-70m",  # 96M parameters | English
    "pythia-1.4b": "EleutherAI/pythia-1.4b",  # 2B parameters | English
    "pythia-6.9b": "EleutherAI/pythia-6.9b",  # 7B parameters | English
}

_whitespace = {
    "Llama-3.2-1B": "Ġ",
    "gpt2": "Ġ",
    "bloom-1b1": "Ġ",
    "bloom-1b7": "Ġ",
    "opt-125m": "Ġ",
    "opt-1.3b": "Ġ",
    "falcon-7b": "Ġ",  # TODO
    "falcon-40b": "Ġ",  # TODO
    "pythia-70m": "Ġ",
    "pythia-1.4b": "Ġ",
    "pythia-6.9b": "Ġ",
}


def load_llm(model_key: str = "Llama") -> tuple[nn.Module, PreTrainedTokenizerBase, str]:
    """Load a language model, tokenizer, and the model-specific whitespace character.

    Args:
        model_key (str): Key identifying which model to load.
    Returns:
        tuple:
            - nn.Module: The loaded language model on the appropriate device.
            - PreTrainedTokenizerBase: The corresponding tokenizer.
            - str: The whitespace character used by the model's tokenizer.
    """
    global models, _whitespace

    if model_key not in models:
        raise Exception(f"Model not supported: {model_key}")

    model_name = models[model_key]
    whitespace = _whitespace[model_key]

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    return model, tokenizer, whitespace
