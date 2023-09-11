import torch


ARCHITECTURE_DEFAULT_DTYPE = "float16" if torch.backends.mps.is_available() else "bfloat16"
ARCHITECTURE_DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
