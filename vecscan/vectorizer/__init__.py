import logging
import os
import time
import traceback
from typing import IO, List, Union

import tqdm

import torch
from safetensors.torch import _tobytes


logger = logging.getLogger("vecscan")


class Vectorizer:
    """An abstract class for extracting embedding vectors from transformer models
    Args:
        vectorizer_type (str): a value in [`openai_api`, `bert_cls`, `sbert`]
        model_path (str): path for the model directory or model name in API
        batch_size (int): batch size
        vec_dim (int): vector dimension
        vec_dtype (str): dtype string of output Tensor
            - default: `bfloat16`
        kwargs: not used
    """
    @classmethod
    def create(cls, vectorizer_type: str, model_path: str, vec_dtype: str="bfloat16", **kwargs):
        """Create an instance of Vectorizer implementation class
        Args:
            vectorizer_type (str): a value in [`openai_api`, `bert_cls`, `sbert`]
            model_path (str): path for the model directory or model name in API
            vec_dtype (str): dtype string of output Tensor
                - default: `bfloat16`
            kwargs: keyword arguments for each Vectorizer implementation class
        Returns:
            Vectorizer: new instance of Vectorizer implementation class
        """
        vectorizer_cls = {
            "openai_api": VectorizerOpenAIAPI,
            "bert_cls": VectorizerBertCLS,
            "sbert": VectorizerSBert,
        }[vectorizer_type]
        vectorizer = vectorizer_cls(vectorizer_type=vectorizer_type, model_path=model_path, vec_dtype=vec_dtype, **kwargs)
        return vectorizer

    def __init__(
        self,
        vectorizer_type: str,
        model_path: str,
        batch_size: int,
        vec_dim: int,
        vec_dtype: str="bfloat16",
        device: Union[str, torch.device]=None,
        **kwargs,
    ):
        self.vectorizer_type = vectorizer_type
        self.model_path = model_path
        self.batch_size = batch_size
        self.vec_dim = vec_dim
        self.vec_dtype = getattr(torch, vec_dtype) if isinstance(vec_dtype, str) else vec_dtype
        self.device = assign_device(device)

    def vectorize(self, batch: List[str]) -> torch.Tensor:
        """Prototype method for extracting embedding vectors
        Args:
            `batch`: text list to embed
                - type: `List[str]`
        Returns:
            torch.Tensor: embedding results
                - shape: [len(batch), dim]
                - dtype: vec_dtype
                - device: depends on Vectorizer implementation class
        """
        raise NotImplementedError()

    def vectorize_file(self, fin: IO, fout: IO) -> int:
        """Extract embedding vectors for file
        Args:
            fin (IO): text input
            fout (IO): binary output
        Returns:
            int: number of embedded lines
        """
        def _exec(batch: List[str]):
            vectors = self.vectorize(batch).cpu()
            byte_buf = _tobytes(vectors, "dummy")
            fout.write(byte_buf)

        batch = [None] * self.batch_size
        vec_count = 0
        for line in tqdm.tqdm(fin):
            line = line.rstrip("\n")
            batch[vec_count % self.batch_size] = line
            vec_count += 1
            if vec_count % self.batch_size == 0:
                _exec(batch)
        if vec_count % self.batch_size > 0:
            _exec(batch[:vec_count % self.batch_size])
        return vec_count

    def _make_info(self, vec_count: int) -> dict:
        return {
            "vec_dim": self.vec_dim,
            "vec_count": vec_count,
            "vec_dtype": str(self.vec_dtype).replace("torch.", ""),
            "vectorizer_type": self.vectorizer_type,
            "model_path": self.model_path,
        }


class VectorizerOpenAIAPI(Vectorizer):
    """Obtain text embedding using OpenAI API"""
    def __init__(
        self,
        model_path: str,
        batch_size: int=256,
        vec_dim: int=1536,
        max_retry: int=3,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            batch_size=batch_size,
            vec_dim=vec_dim,
            **kwargs,
        )
        try:
            import openai
        except Exception as e:
            logger.error("You need to install openai package to use VectorizerOpenAIAPI.")
            raise e
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.openai_embedding_create = openai.Embedding.create
        self.max_retry = max_retry

    def vectorize(self, batch: List[str]) -> torch.tensor:
        e = None
        for _ in range(self.max_retry):
            try:
                vectors = [_["embedding"] for _ in self.openai_embedding_create(input=batch, model=self.model_path)["data"]]
                return torch.tensor(vectors, dtype=self.vec_dtype)
            except Exception as e:
                logger.warning(traceback.format_exc())
                if _ + 1 == self.max_retry:
                    raise e
                logger.warning("waiting for retry")
                time.sleep(3)


class VectorizerBertCLS(Vectorizer):
    """Extracts [CLS] embeddings for the specified `hidden_layer` from BERT-style transformer.
    - default: `hidden_layer=3`
    - In [Self-Guided Contrastive Learning for BERT Sentence Representations](https://arxiv.org/abs/2106.07345),
    they reported that the performance of [CLS] token embedding varies greatly depending on the layer to use.
    """
    def __init__(
        self,
        model_path: str,
        batch_size: int=512,
        vec_dim: int=768,
        max_length: int=512,
        hidden_layer: int=3,
        normalize: bool=True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            batch_size=batch_size,
            vec_dim=vec_dim,
            **kwargs,
        )
        self.max_length = max_length
        self.hidden_layer = hidden_layer
        self.normalize = normalize
        try:
            from transformers import BertJapaneseTokenizer, BertForSequenceClassification
        except Exception as e:
            logger.error("You need to install transformers package to use VectorizerBertCLS.")
            raise e
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path).half().to(self.device)

    def vectorize(self, batch: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(batch, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt').to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            batch_vectors = outputs["hidden_states"][self.hidden_layer].transpose(0, 1)[0]
            if self.normalize:
                batch_vectors = torch.nn.functional.normalize(batch_vectors)
            return batch_vectors.to(self.vec_dtype)


class VectorizerSBert(Vectorizer):
    """Extracts embeddings from `sentence-transformers`"""
    def __init__(
        self,
        model_path: str,
        batch_size: int=512,
        vec_dim: int=768,
        normalize: bool=True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            batch_size=batch_size,
            vec_dim=vec_dim,
            **kwargs,
        )
        self.normalize = normalize
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            logger.error("You need to install sentence-transformers package to use VectorizerSBert.")
            raise e
        self.model = SentenceTransformer(self.model_path, device=self.device)

    def vectorize(self, batch: List[str]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch_vectors = self.model.encode(batch, convert_to_tensor=True)
            if self.normalize:
                batch_vectors = torch.nn.functional.normalize(batch_vectors)
            return batch_vectors.to(self.vec_dtype)


def assign_device(device: Union[str, torch.device]) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device
