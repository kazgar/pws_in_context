import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

models = {
    "Llama": "meta-llama/Llama-3.2-1B",
    "GPT2": "openai-community/gpt2",
}

_whitespace = {"Llama": "Ġ", "GPT2": None}  # TODO


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
