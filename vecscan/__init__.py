import torch

from .scanner import VectorScanner, Similarity
from .vectorizer import Vectorizer
from .vector_loader import VectorLoader


ARCHITECTURE_DEFAULT_DTYPE = "float16" if torch.backends.mps.is_available() else "bfloat16"


__all__ = ["VectorScanner", "Similarity", "Vectorizer", "VectorLoader", "ARCHITECTURE_DEFAULT_DTYPE"]
