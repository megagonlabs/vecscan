from enum import Enum
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
import safetensors


logger = logging.getLogger("vecscan")


SAFETENSORS_NUM_SHARDS_FIELD = "num_shard"
SAFETENSORS_SHARD_NAME_FORMAT = "shard_{}"


class Similarity(Enum):
    """Enum for similarity functions"""
    def _dot(m: Tensor, v: Tensor) -> Tensor:
        if m.device.type == "cpu":
            if m.numel() <= 4096:  # float casting
                if m.dtype == torch.bfloat16:
                    return torch.mv(m.float(), v.float()).bfloat16()
                elif m.dtype == torch.float16:
                    return torch.mv(m.float(), v.float()).float16()
        return torch.mv(m, v)
    Dot = _dot
    """Dot product similarity (assuming all the vectors normalized to have the |v|=1)
    - This implementation avoids the accuracy degradation in `torch.mv()` that occurs when the number of elements in the bfloat16 or float16 matrix on CPUs is 4096 or less
    """
    Cosine = lambda m, v: torch.nn.functional.cosine_similarity(m, v, dim=1)
    """Cosine similarity using `torch.nn.functional.cosine_similarity()`
    """
    L1 = lambda m, v: (m - v).abs().sum(dim=1)
    """L1 norm (Manhattan distance) using `(m - v).abs().sum(dim=1)` (`norm(ord=1)` has a problem of accuracy degradation for bfloat16 and float16)
    """
    L2 = lambda m, v: torch.linalg.norm(m - v, dim=1, ord=2)
    """L2 norm (Euclidean distance) using `torch.linalg.norm(m - v, dim=1, ord=2)`
    """


def _info(shards: Union[Tensor, List[Tensor]]) -> str:
    if isinstance(shards, Tensor):
        return f"({list(shards.shape)} {shards.dtype} {shards.device})"
    else:
        shape = list(shards[0].shape)
        shape[0] = sum(len(_) for _ in shards)
        return f"({shape} {shards[0].dtype} {shards[0].device})"


class VectorScanner:
    """An implementation for dense vector linear search engine
    Args:
        shards (List[torch.Tensor]): a List of 2d Tensor instances which stores the search target dense vectors
            - shape: all elements must have the same dim (size of the element must be < 2**32 bytes)
            - dtype: all elements must have the same dtype
            - device: all elements must have the same device
    """
    def __init__(self, shards: List[Tensor]):
        self.shards = shards
        self.offsets = [0]
        for _ in self.shards:
            self.offsets.append(self.offsets[-1] + len(_))

    def __len__(self) -> int:
        """
        Returns:
            int: number of total rows in `self.shards`
        """
        return self.offsets[-1]

    def __getitem__(self, index: int) -> Tensor:
        """
        Args:
            index (int): index for all rows in `self.shards`
        Returns:
            torch.Tensor: a row vector in `self.shards`
        """
        prev_offset = 0
        for shard, offset in zip(self.shards, self.offsets[1:]):
            if index < offset:
                return shard[index - prev_offset]
            prev_offset = offset
        else:
            raise Exception(f"Bad index {index} >= {self.offsets[-1]}")

    @property
    def dtype(self):
        """
        Returns:
            dtype of Tensor instances in `self.shards`
        """
        return self.shards[0].dtype

    @property
    def device(self):
        """
        Returns:
            device of Tensor instances in `self.shards`
        """
        return self.shards[0].device

    @property
    def shape(self):
        """
        Returns:
            shape of entire vectors in `self.shards`
        """
        shape = list(self.shards[0].shape)
        shape[0] = len(self)
        return torch.Size(shape)

    def to(self, dst: Any):
        """apply to(dst) for all the Tensor instances in `self.shards`
        Args:
            dst: dtype or device
        Returns:
            self
        """
        for _ in self.shards:
            _.to(dst)
        return self

    def score(self, query_vector: Tensor, similarity_func: Callable[[Tensor, Tensor], Tensor]=Similarity.Dot) -> Tensor:
        """Calculate the similarity scores between vectors and `query_vector` by using `similarity_func`
        Args:
            query_vector (torch.Tensor): a dense 1d Tensor instance which stores the embedding vector of query text
                - shape: [dim]
                - dtype: same as the elements of `self.shards`
                - device: same as the elements of `self.shards`
            similarity_func (Callable[[Tensor, Tensor], Tensor]): a Callable which calculates the similarities between target dense matrix and query vector  
                - default: `Similarity.Dot` - Dot product similarity (assuming all the vectors normalized to have the |v|=1)
                - shapes: arg1=[records, dim], arg2=[dim], return=[records]
                - dtype: same as the elements of `self.shards`
                - device: same as the elements of `self.shards`
        Returns:
            torch.Tensor: a dense 1d Tensor instance which stores the similarity scores
                - shape: [records]
                - dtype: same as the elements of `self.shards`
                - device: same as the elements of `self.shards`
        """
        logger.debug(f"start score(), query_vector={_info(query_vector)}, {similarity_func=}")
        query_vector = query_vector.to(self.shards[0].dtype).to(self.shards[0].device)
        result = torch.cat(list(similarity_func(_, query_vector) for _ in self.shards))
        logger.debug(f"end   score(), result={_info(result)}")
        return result

    def search(self, query_vector: Tensor, target_ids: Optional[Union[List[int], Tensor]]=None, top_n: Optional[int]=1000, similarity_func: Callable[[Tensor, Tensor], Tensor]=Similarity.Dot) -> Tuple[Tensor, Tensor]:
        """Sort the result of `score()` and then apply `target_ids` filter
        Args:
            query_vector (torch.Tensor): a dense 1d Tensor instance which stores the embedding vector of query text
                - shape: [dim]
                - dtype: same as the elements of `self.shards`
                - device: same as the elements of `self.shards`
            target_ids (Optional[Union[List[int], Tensor]]): search target is limited to records included in target_ids if specified
                - default: None
            top_n (Optional[int]): search result list is limited to top_n if specified
                - default: `1000`
            similarity_func (Callable[[Tensor, Tensor], Tensor]): a Callable which calculates the similarities between target dense matrix and query vector  
                - default: `Similarity.Dot` - Dot similarity (assuming all the vectors normalized to have the |v|=1)
                - shapes: arg1=[records, dim], arg2=[dim], return=[records]
                - dtype: same as the elements of `self.shards`
                - device: same as the elements of `self.shards`
        Returns:
            Tuple[Tensor, Tensor]: a Tuple which contains search results (sorted_scores, doc_ids)
                - shapes: [top_n] or [records]
                - dtypes: sorted_scores=`self.shards[0].dtype`, doc_ids=`torch.int64`
                - device: same as the elements of `self.shards`
        """
        scores = self.score(query_vector, similarity_func)
        logger.debug(f"start search(), len(target_ids)={'None' if target_ids is None else len(target_ids)}, {top_n=}")
        if target_ids is not None:
            if isinstance(target_ids, list):
                target_ids = torch.tensor(target_ids, dtype=torch.int64, device=self.shards[0].device)
            scores = torch.gather(scores, 0, target_ids)
        if top_n:
            doc_ids = torch.argsort(scores, descending=True)[:top_n]
            sorted_scores = torch.gather(scores, 0, doc_ids)
        else:
            sorted_scores, doc_ids = torch.sort(scores, descending=True)
        if target_ids is not None:
            doc_ids = torch.gather(target_ids, 0, doc_ids)
        logger.debug(f"end   search(), sorted_scores={_info(sorted_scores)}, doc_ids={_info(doc_ids)}")
        return sorted_scores, doc_ids

    def save_file(self, path: str):
        """Save `self.shards` to new safetensors file
        Args:
            path (str): path for new safetensors file
        """
        logger.debug(f"start save_file(), {path=}")
        tensors = {SAFETENSORS_NUM_SHARDS_FIELD: torch.tensor(len(self.shards), dtype=torch.int64)}
        for _, shard in enumerate(self.shards):
            tensors[SAFETENSORS_SHARD_NAME_FORMAT.format(_)] = shard
        safetensors.torch.save_file(tensors, path)
        logger.debug(f"end   save_file()")

    @classmethod
    def load_file(cls, path: str, device: Optional[str]=None, normalize: bool = False, break_in: bool=True):
        """Create VectorScanner instance and load 2d tensors to `self.shards` from safetensors file
        Args:
            path (str): path for safetensors file to load
            device (Optional[str]): a device to load vectors (typically `cpu`, `cuda`, or `mps`)
            normalize (bool): normalize the norm of each vector if True
            break_in (bool): execute break-in run after loading entire vectors
        Returns:
            VectorScanner: new VectorScanner instance
        """
        device = VectorScanner._assign_device(device)
        logger.debug(f"start load_file(): {path=}, {device=}, {normalize=}, {break_in=}")
        tensors = safetensors.torch.load_file(path, device=device)
        num_shards = tensors[SAFETENSORS_NUM_SHARDS_FIELD]
        shards = [tensors[SAFETENSORS_SHARD_NAME_FORMAT.format(_)] for _ in range(num_shards)]
        if normalize:
            shards = [torch.nn.functional.normalize(_) for _ in shards]
        elif break_in:
            for _ in shards:
                torch.mv(_, _[0])
        logger.debug(f"end   load_file(): {_info(shards)}")
        return VectorScanner(shards)

    @staticmethod
    def _assign_device(device: str) -> str:
        if device is not None:
            return device
        elif torch.cuda.is_available():
            return "cuda" 
        elif torch.backends.mps.is_available():
            return "mps" 
        else:
            return "cpu"
