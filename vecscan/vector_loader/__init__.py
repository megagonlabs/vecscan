import json
import logging
import re
import sys
from typing import Any, IO, Optional

import numpy
import torch
from torch import Tensor

from .. import VectorScanner


logger = logging.getLogger("vecscan")


class VectorLoader:
    """An abstract class for loading embedding vectors from file
    Args:
        vec_dim (Optional[int]): vector dimension
            - default: None - determine from inputs
        normalize (bool): normalize all the vectors to have |v|=1 if True
            - default: False
        safetensors_dtype (Any): dtype of output Tensor instances
            - default: "bfloat16"
        shard_size (int): maximum size of each shard in safetensors file
            - default: `2**32` (in byte)
        kwargs: keyword arguments for `VectorLoader` implementation class
    """
    @classmethod
    def create(cls, input_format: str, vec_dim: Optional[int]=None, normalize: bool=False, safetensors_dtype: Any="bfloat16", shard_size: int=2**32, **kwargs):
        """Create an instance of VectorLoader implementation class
        Args:
            input_format (str): a value in [`csv`, `jsonl`, `binary`]
            vec_dim (Optional[int]): vector dimension
                - default: None - determine from inputs
            normalize (bool): normalize all the vectors to have |v|=1 if True
                - default: False
            safetensors_dtype (Any): dtype of output Tensor instances
                - default: "bfloat16"
            shard_size (int): maximum size of each shard in safetensors file
                - default: `2**32` (in byte)
            kwargs: keyword arguments for `VectorLoader` implementation class
        Returns:
            VectorLoader: new instance of `VectorLoader` implementation class
        """
        vector_loader_cls = {
            "csv": CsvVectorLoader,
            "jsonl": JsonlVectorLoader,
            "binary": BinaryVectorLoader,
        }[input_format]
        vector_loader = vector_loader_cls(
            vec_dim=vec_dim,
            normalize=normalize,
            safetensors_dtype=safetensors_dtype,
            shard_size=shard_size,
            **kwargs,
        )
        return vector_loader

    def __init__(self, vec_dim: Optional[int]=None, normalize: bool=False, safetensors_dtype: Any="bfloat16", shard_size: int=2**32, **kwargs):
        self.vec_dim = vec_dim
        self.safetensors_dtype = getattr(torch, safetensors_dtype) if isinstance(safetensors_dtype, str) else safetensors_dtype
        self.normalize = normalize
        self.shard_size = shard_size
        self.max_records_in_shard = None
        self.file: IO = None

    def create_vector_scanner(self) -> VectorScanner:
        """Creates a `VectorScanner` instance using the Tensor read from the input.
        Returns:
            VectorScanner: new `VectorScanner` instance
        """
        shards = []
        while True:
            shard = self.load_shard()
            if shard is None:
                break
            shards.append(shard)
        scanner = VectorScanner(shards)
        return scanner

    def load_shard(self, fin: IO=sys.stdin) -> Optional[Tensor]:
        """Prototype method for loading single shard from input
        Returns:
            Optional[Tensor]: a Tensor instance if one or more records exists, None for end of file
        """
        raise Exception("not implemented")


class CsvVectorLoader(VectorLoader):
    def __init__(self, skip_first_line: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_first_line = skip_first_line
        self.first_line_skipped = False
        logger.debug(f"CsvVectorLoader: vec_dim={self.vec_dim}, safetensors_dtype={self.safetensors_dtype}, skip_first_line={self.skip_first_line}")

    def load_shard(self, fin: IO=sys.stdin) -> Optional[Tensor]:
        if self.skip_first_line and not self.first_line_skipped:
            line = fin.readline().strip('" \n')
            self.first_line_skipped = True
            logger.info(f"skip first line: {line}")
        vectors = []
        while True:
            line = fin.readline()
            if not line:
                break
            target = [float(_) for _ in re.split(r'"? *, *"?', line.rstrip("\n"))]
            vector = torch.tensor(target, dtype=self.safetensors_dtype)
            if self.vec_dim is None:
                self.vec_dim = len(vector)
            if self.max_records_in_shard is None:
                self.max_records_in_shard = _max_records_in_shard(self.vec_dim, self.safetensors_dtype, self.shard_size)
                logger.info(f"vec_dim={self.vec_dim}")
            assert self.vec_dim == len(vector), f"vec_dim:{self.vec_dim} != len(vector):{len(vector)}"
            vectors.append(vector)
            if len(vectors) == self.max_records_in_shard:
                break
        if vectors:
            shard = torch.stack(vectors)
            if self.normalize:
                shard = torch.nn.functional.normalize(shard)
            return shard
        else:
            return None


class JsonlVectorLoader(VectorLoader):
    def __init__(self, target_field: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)
        self.target_field = target_field
        logger.debug(f"JsonlVectorLoader: vec_dim={self.vec_dim}, target_field={self.target_field}, safetensors_dtype={self.safetensors_dtype}")

    def load_shard(self, fin: IO=sys.stdin) -> Optional[Tensor]:
        vectors = []
        while True:
            line = fin.readline()
            if not line:
                break
            record = json.loads(line)
            if self.target_field is None:
                target = record
            else:
                target = record[self.target_field]
            vector = torch.tensor(target, dtype=self.safetensors_dtype)
            if self.vec_dim is None:
                self.vec_dim = len(vector)
            if self.max_records_in_shard is None:
                self.max_records_in_shard = _max_records_in_shard(self.vec_dim, self.safetensors_dtype, self.shard_size)
                logger.info(f"vec_dim={self.vec_dim}")
            assert self.vec_dim == len(vector), f"vec_dim:{self.vec_dim} != len(vector):{len(vector)}"
            vectors.append(vector)
            if len(vectors) == self.max_records_in_shard:
                break
        if vectors:
            shard = torch.stack(vectors)
            if self.normalize:
                shard = torch.nn.functional.normalize(shard)
            return shard
        else:
            return None


class BinaryVectorLoader(VectorLoader):
    def __init__(self, input_dtype: str="float32", **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.vec_dim, int), "specify vec_dim with int value"
        self.input_dtype = getattr(numpy, input_dtype) if isinstance(input_dtype, str) else input_dtype
        self.max_records_in_shard = _max_records_in_shard(self.vec_dim, self.safetensors_dtype, self.shard_size)
        logger.debug(f"BinaryVectorLoader: vec_dim={self.vec_dim}, input_dtype={self.input_dtype}, safetensors_dtype={self.safetensors_dtype}")

    def load_shard(self, fin: IO=sys.stdin.buffer) -> Optional[Tensor]:
        if self.max_records_in_shard > 0:
            nparray = numpy.fromfile(fin, self.input_dtype, self.max_records_in_shard * self.vec_dim)
        else:
            nparray = numpy.fromfile(fin, self.input_dtype)
        if len(nparray) > 0:
            shard = torch.tensor(nparray, dtype=self.safetensors_dtype).reshape(-1, self.vec_dim)
            if self.normalize:
                shard = torch.nn.functional.normalize(shard)
            return shard
        else:
            return None


def _max_records_in_shard(vec_dim: int, torch_dtype: Any, shard_size: int):
    element_size = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
    }[torch_dtype]
    return shard_size // (element_size * vec_dim + 8)
