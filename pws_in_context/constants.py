from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)
