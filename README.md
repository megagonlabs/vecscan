# vecscan
`vecscan`: A Linear-scan-based High-speed Dense Vector Search Engine

## Introduction

The vecscan is a dense vector search engine that performs similarity search for embedding databases in linear and greedy way by using the SIMDs (such as AVX2, AVX512, or AMX), CUDA, or MPS through PyTorch. (Note that using a GPU is super fast, but not required, and modern CPUs are more cost effective.)

The vecscan employs simple linear-scan based algorithms, and it does not cause quantization errors that tend to be a problem with approximate neighborhood searches like `faiss`. The vecscan makes it very easy to build your vector search applications.

In vecscan, the default dtype of embedding vectors is `torch.bfloat16` (you can use `torch.float16` for Apple's MPS and CUDA devices, or `torch.float32` for any devices instead), and the file format of embedding database is `safetensors`. The embedding database, which holds 1 million records of 768-dimensional bfloat16 or float16 vectors, occupies 1.5GB of main memory (or GPU memory). If you're using 8x Sapphire Rapids vCPUs, `VectorScanner.search()` will take only 0.1[sec] for similarity score calculation and sorting (1M-records, 768-dim, bfloat16). The benchmarks for major CPUs and GPUs can be found in [Benchmarks](#benchmarks) section.

### Recommended Environment

- Intel Architecture
  - Best
    - AMX on bfloat16
      - Sapphire Rapids
  - Better
    - AVX512_BF16 on bfloat16
      - Cooper Lake, 4th gen EPYC.
      - The CPUs of GCP N2 instance are Cascade Lake or later in its specification but actually Cooper Lake appeared in our benchmark experiments
  - Limited
    - AVX512F/AVX2 on float32 (bfloat16 is too slow)
      - Consumer CPUs or older Xeon CPUs
- Apple Silicon
  - Best
    - MPS on float16
  - Limited
    - Without MPS, only float32 is available
- GPUs (Optional)
  - Best
    - Ampere or later on bfloat16
      - L4(24GB), and L40(48GB) are the best choice for cost performance in 2023
    - Volta or later on float16
      - T4(16GB)

### Preparation

#### Install from PyPI

```console
$ pip install vecscan
```

#### Install from Repository

```console
$ git clone https://github.com/megagonlabs/vecscan.git
$ cd vecscan
$ pip install -e .
```

#### For Specific CUDA Version

If you're using GPUs with CUDA, install torch with CUDA version by using "--index-url" options.

Example for CUDA 11.8:
```console
$ pip install -U torch --index-url https://download.pytorch.org/whl/cu118
```

### How to Search

The latency and the throughput of `VectorScanner.search()` fully depend on the total FLOPs of the processors.
We recommend you to use the latest XEON platform (such as GCP C3 instance which supports AMX), a MPS device, or a CUDA GPU device (such as NVIDIA L4) with enough memory to load entire safetensors vector file.

If you're using the OpenAI API for embedding, you need to install `openai` package and set your api key to the environmental variable beforehand. See [Embedding Examples](#embed-text-by-openais-text-embedding-ada-002) section for details.

```Python
from vecscan import VectorScanner, Vectorizer

# load safetensors file
scanner = VectorScanner.load_file("path_to_safetensors")
# for Apple MPS:
# scanner = VectorScanner.load_file("path_to_safetensors", device="mps")

# use OpenAI's text-embedding-ada-002 with the environmental variable "OPENAI_API_KEY" 
vectorizer = Vectorizer.create(vectorizer_type="openai_api", model_path="text-embedding-ada-002")
# for float16:
# vectorizer = Vectorizer.create(vectorizer_type="openai_api", model_path="text-embedding-ada-002", vec_dtype="float16")

# get query embedding
query_vec = vectorizer.vectorize(["some query text"])[0]
# execute search and get similarity scores and corresponding document ids in descendant order
sim_scores, doc_ids = scanner.search(query_vec)
```

#### Notice for MPS

Although the conditions have not been determined, when calling `VectorScanner.score()` or `VectorScanner.search()` with `query_vec` which comes from a row near the end of the 2d matrix in mps, the process may be forcibly terminated with an error like `error: the range subRange.start + subRange.length does not fit in dimension[1]`.
In such cases, you can avoid the error by cloning the Tensor as follows:

```Python
sim_scores, doc_ids = scanner.search(query_vec.clone().detach())
```

## APIs

### Class Structures

```
vecscan -/- scanner.py ----+- VectorScanner
         |                 |
         |                 +- Similarity (Enum)
         |                     +- Dot    # Dot product similarity (assuming all the vectors normalized to have the |v|=1)
         |                     +- Cosine # cosine similarity (general purpose)
         |                     +- L1     # Manhattan distance
         |                     +- L2     # Euclidean distance
         |
         /- vector_loader -/- VectorLoader (`convert_to_safetensors` command)
         |                     +- CsvVectorLoader
         |                     +- JsonlVectorLoader
         |                     +- BinaryVectorLoader
         |
         /- vectorizer ----/- Vectorizer (`vectorize` command)
                               +- VectorizerOpenAIAPI
                               +- VectorizerBertCLS
                               +- VectorizerSBert
```

### VectorScanner

An implementation for dense vector linear search engine

#### Import

```Python
from vecscan import VectorScanner
```

#### Class Method

- `VectorScanner.load_file(cls, path, device=None, normalize=False, break_in=True)`
  - Create VectorScanner instance and load 2d tensors to `self.shards` from safetensors file
  - Args:
    - path (str): path for safetensors file to load
    - device (Optional[str]): a device to load vectors (typically `cpu`, `cuda`, or `mps`)
    - normalize (bool): normalize the norm of each vector if True
    - break_in (bool): execute break-in run after loading entire vectors
  - Returns:
    - VectorScanner: new VectorScanner instance

#### Instance Methods

- `VectorScanner(shards)`
  - Args:
    - shards (List[torch.Tensor]): a List of 2d Tensor instances which stores the search target dense vectors
      - shape: all elements must be the same dim (size of the element must be < 2**32 bytes)
      - dtype: all elements must be the same dtype
      - device: all elements must be on the same device
- `score(self, query_vector, similarity_func=Similarity.Dot)`
  - Calculate the similarity scores between vectors and `query_vector` by using `similarity_func`
  - Args:
    - query_vector (torch.Tensor): a dense 1d Tensor instance which stores the embedding vector of query text
      - shape: [dim]
      - dtype: same as the elements of `self.shards`
      - device: same as the elements of `self.shards`
    - similarity_func (Callable[[Tensor, Tensor], Tensor]): a Callable which calculates the similarities between target dense matrix and query vector  
      - default: `Similarity.Dot` - Dot product similarity (assuming all the vectors normalized to have the |v|=1)
      - shapes: arg1=[records, dim], arg2=[dim], return=[records]
      - dtype: same as the elements of `self.shards`
      - device: same as the elements of `self.shards`
  - Returns:
    - torch.Tensor: a dense 1d Tensor instance which stores the similarity scores
    - shape: [records]
    - dtype: same as the elements of `self.shards`
    - device: same as the elements of `self.shards`
- `search(self, query_vector, target_ids=None, n_best=1000, similarity_func=Similarity.Dot)`
  - Sort the result of `score()` and then apply `target_ids` filter
  - Args:
    - query_vector (torch.Tensor): a dense 1d Tensor instance which stores the embedding vector of query text
      - shape: [dim]
      - dtype: same as the elements of `self.shards`
      - device: same as the elements of `self.shards`
    - target_ids (Optional[Union[List[int], Tensor]]): search target is limited to records included in target_ids if specified
      - default: None
    - top_n (Optional[int]): search result list is limited to top_n if specified
      - default: `1000`
    - similarity_func (Callable[[Tensor, Tensor], Tensor]): a Callable which calculates the similarities between target dense matrix and query vector  
      - default: `Similarity.Dot` (Dot product similarity (assuming all the vectors normalized to have the |v|=1)
      - shapes: arg1=[records, dim], arg2=[dim], return=[records]
      - dtype: same as the elements of `self.shards`
      - device: same as the elements of `self.shards`
  - Returns:
    - Tuple[Tensor, Tensor]: a Tuple which contains search results (sorted_scores, doc_ids)
    - shapes: [top_n] or [records]
    - dtypes: sorted_scores=`self.shards[0].dtype`, doc_ids=`torch.int64`
    - device: same as the elements of `self.shards`
- `save_file(self, path)`
  - Save vectors to new safetensors file
  - Args:
    - path (str): path for new safetensors file
- `__len__(self)`
  - Returns:
    - int: number of total rows in `self.shards`
- `__getitem__(self, index: int)`
  - Args:
    - index (int): index for all rows in `self.shards`
  - Returns:
    - Tensor: a row vector in `self.shards`
    - shapes: [dim]
    - dtype: same as the elements of `self.shards`
    - device: same as the elements of `self.shards`
- `dtype(self)`
  - Returns:
    - dtype of Tensor instances in `self.shards`
- `device(self)`
  - Returns:
    - device of Tensor instances in `self.shards`
- `shape(self)`
  - Returns:
    - shape of entire vectors in `self.shards`
- `to(self, dst: Any)`
  - apply to(dst) for all the Tensor instances in `self.shards`
  - Args:
    - dst: dtype or device
  - Returns:
    self

### Similarity

Enum for similarity functions.

#### Import

```Python
from vecscan import Similarity
```

#### Enum Entries

- `Similarity.Dot`
  - Dot product similarity (assuming all the vectors normalized to have the |v|=1)
  - This implementation avoids the accuracy degradation in `torch.mv()` that occurs when the number of elements in the bfloat16 or float16 matrix on CPUs is 4096 or less
- `Similarity.Cosine`
  - Cosine similarity using `torch.nn.functional.cosine_similarity()`
- `Similarity.L1`
  - L1 norm (Manhattan distance) using `(m - v).abs().sum(dim=1)` (`norm(ord=1)` has a problem of accuracy degradation for bfloat16 and float16)
- `Similarity.L2`
  - L2 norm (Euclidean distance) using `torch.linalg.norm(m - v, dim=1, ord=2)`

### VectorLoader

An abstract class for loading embedding vectors from file

#### Import

```Python
from vecscan import VectorLoader
```

#### Class Method

- `create(cls, input_format, vec_dim, normalize, safetensors_dtype, shard_size, **kwargs)`
  - Create an instance of VectorLoader implementation class
  - Args:
    - input_format (str): a value in [`csv`, `jsonl`, `binary`]
    - vec_dim (Optional[int]): vector dimension
      - default: None - determine from inputs
    - normalize (bool): normalize all the vectors to have |v|=1 if True
      - default: False
    - safetensors_dtype (Any): dtype of output Tensor instances
      - default: "bfloat16"
    - shard_size (int): maximum size of each shard in safetensors file
      - default: `2**32` (in byte)
    - kwargs: keyword arguments for `VectorLoader` implementation class
  - Returns:
    - VectorLoader: new instance of `VectorLoader`` implementation class

#### Instance Methods

- `VectorLoader(input_format, vec_dim, normalize, safetensors_dtype, shard_size, **kwargs)`
  - Constructor
  - Args:
    - vec_dim (Optional[int]): vector dimension
      - default: None - determine from inputs
    - normalize (bool): normalize all the vectors to have |v|=1 if True
      - default: False
    - safetensors_dtype (Any): dtype of output Tensor instances
      - default: "bfloat16"
    - shard_size (int): maximum size of each shard in safetensors file
      - default: `2**32` (in byte)
    - kwargs: keyword arguments for `VectorLoader` implementation class
- `create_vector_scanner(self)`
  - Creates a `VectorScanner` instance using the Tensor read from the input.
  - Returns:
    - VectorScanner: new `VectorScanner` instance
- `load_shard(self)`
  - Prototype method for loading single shard from input
  - Returns:
    - Optional[Tensor]: a Tensor instance if one or more records exists, None for end of file

#### VectorLoader Implementations

- `CsvVectorLoader`
  - Converts CSV lines to safetensors
  - Skips first line if `skip_first_line` is True
- `JsonlVectorLoader`
  - Converts JSONL lines to safetensors
  - default: each line has a list element consists of float values
  - specify `target_field` if each line has a dict element and the embedding field is directly under the root dict
- `BinaryVectorLoader`
  - Converts binary float values to safetensors
  - `vec_dim` and `input_dtype` must be specified to determine the byte length of a row

#### Run `convert_to_safetensors` command

You can convert existing embedding data to vecscan's safetensors file by running `convert_to_safetensors` command.

```console
$ convert_to_safetensors -h

usage: convert_to_safetensors [-h] -f {csv,jsonl,binary} -o OUTPUT_SAFETENSORS_PATH [-d VEC_DIM] [-n] [-s] [-t TARGET_FIELD] [--input_dtype INPUT_DTYPE] [--safetensors_dtype SAFETENSORS_DTYPE] [--shard_size SHARD_SIZE] [-v]

A tool converting vector data to a safetensors file.

optional arguments:
  -h, --help            show this help message and exit
  -f {csv,jsonl,binary}, --input_format {csv,jsonl,binary}
  -o OUTPUT_SAFETENSORS_PATH, --output_safetensors_path OUTPUT_SAFETENSORS_PATH
  -d VEC_DIM, --vec_dim VEC_DIM
  -n, --normalize
  -s, --skip_first_line
  -t TARGET_FIELD, --target_field TARGET_FIELD
  --input_dtype INPUT_DTYPE
  --safetensors_dtype SAFETENSORS_DTYPE
  --shard_size SHARD_SIZE
  -v, --verbose
```

#### Converting Examples

##### Convert CSV file to safetensors

Output `bfloat16` safetensors (for cpu or cuda):

```console
$ cat << EOS > sample.csv
1.0, 0.0, 0.0, 0.0
0.0, 1.0, 0.0, 0.0
EOS

$ convert_to_safetensors -f csv -o sample.csv.safetensors < sample.csv

2023-09-08 01:29:51,613 INFO:vecscan: convert sys.stdin to sample.csv.safetensors
2023-09-08 01:29:51,613 INFO:vecscan:   input_format=csv, input_dtype=float32, vec_dim=None, target_field=None, normalize=False, safetensors_dtype=bfloat16, shard_size=4294967296
2023-09-08 01:29:51,613 INFO:vecscan: 4 records converted
```

Output safetensors for `float16` (for mps or cuda):

```console
$ convert_to_safetensors -f csv -o sample.csv.safetensors --safetensors_dtype float16 < sample.csv
```

If CSV has a title row:

```console
$ convert_to_safetensors -f csv -o sample.csv.safetensors --skip_first_line < sample.csv
```

##### Convert list-style JSONL file to safetensors

```console
$ cat << EOS > sample_list.jsonl
[1.0, 0.0, 0.0, 0.0]
[0.0, 1.0, 0.0, 0.0]
EOS

$ convert_to_safetensors -f jsonl -o sample_list.jsonl.safetensors < sample_list.jsonl

2023-09-08 01:38:47,905 INFO:vecscan: convert sys.stdin to sample_list.jsonl.safetensors
2023-09-08 01:38:47,905 INFO:vecscan:   input_format=jsonl, input_dtype=float32, vec_dim=None, target_field=None, normalize=False, safetensors_dtype=bfloat16, shard_size=4294967296
2023-09-08 01:38:47,906 INFO:vecscan: 2 records converted
```

##### Convert dict-style JSONL file to safetensors

```console
$ cat << EOS > sample_dict.jsonl
{"vec": [1.0, 0.0, 0.0, 0.0]}
{"vec": [0.0, 1.0, 0.0, 0.0]}
EOS

$ convert_to_safetensors -f jsonl -t vec -o sample_dict.jsonl.safetensors < sample_dict.jsonl

2023-09-08 01:41:21,840 INFO:vecscan: convert sys.stdin to sample_dict.jsonl.safetensors
2023-09-08 01:41:21,840 INFO:vecscan:   input_format=jsonl, input_dtype=float32, vec_dim=None, target_field=vec, normalize=False, safetensors_dtype=bfloat16, shard_size=4294967296
2023-09-08 01:41:21,840 INFO:vecscan: 2 records converted
```

##### Convert binary data to safetensors

```console
$ python -c 'import sys; import numpy; m = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], numpy.float32); m.tofile(sys.stdout.buffer)' > sample.vec

$ convert_to_safetensors -f binary -d 4 --input_dtype float32 -o sample.vec.safetensors < sample.vec

2023-09-08 01:50:24,489 INFO:vecscan: convert sys.stdin to sample.vec.safetensors
2023-09-08 01:50:24,489 INFO:vecscan:   input_format=binary, input_dtype=float32, vec_dim=4, target_field=None, normalize=False, safetensors_dtype=bfloat16, shard_size=4294967296
2023-09-08 01:50:24,489 INFO:vecscan: 2 records converted
```

### Vectorizer

An abstract class for extracting embedding vectors from transformer models

#### Import

```Python
from vecscan import Vectorizer
```

#### Class Method

- `Vectorizer.create(cls, vectorizer_type, model_path, vec_dtype, **kwargs)`
  - Creates an instance of Vectorizer implementation class
  - Args:
    - vectorizer_type (str): a value in [`openai_api`, `bert_cls`, `sbert`]
    - model_path (str): path for the model directory or model name in API
    - vec_dtype (str): dtype string of output Tensor
      - default: `bfloat16`
    - kwargs: keyword arguments for each Vectorizer implementation class
  - Returns:
    - Vectorizer: new instance of Vectorizer implementation class

#### Instance Methods

- `Vectorizer(vectorizer_type, model_path, batch_size, vec_dim, vec_dtype, device, **kwargs)`
  - Constructor
  - Args:
    - vectorizer_type (str): a value in [`openai_api`, `bert_cls`, `sbert`]
    - model_path (str): path for the model directory or model name in API
    - batch_size (int): batch size
    - vec_dim (int): vector dimension
    - vec_dtype (str): dtype string of output Tensor
      - default: `bfloat16`
    - kwargs: not used
- `vectorize(self, batch)`
  - Prototype method for extracting embedding vectors
  - Args:
    - `batch`: text list to embed
      - type: `List[str]`
  - Returns:
    - type: `torch.Tensor`
    - shape: [len(batch), dim]
    - dtype: vec_dtype
    - device: depends on Vectorizer implementation class
- `vectorize_file(self, fin, fout)`
  - Extracts embedding vectors for file
  - Args:
    - fin (IO): text input
    - fout (IO): binary output
  - Returns:
    int: number of embedded lines

#### Vectorizer Implementations

- `VectorizerOpenAIAPI`
  - Obtains text embedding using OpenAI API
- `VectorizerBertCLS`
  - Extracts [CLS] embeddings for the specified `hidden_layer` from BERT-style transformer.
    - default: `hidden_layer=3` 
  - In [Self-Guided Contrastive Learning for BERT Sentence Representations](https://arxiv.org/abs/2106.07345), they reported that the performance of [CLS] token embedding varies greatly depending on the layer to use.
- `VectorizerSBERT`
  - Extracts embeddings from `sentence-transformers`

#### Run `vectorize` command

You can embed entire text lines in a file by running `vectorize` command.

```console
$ vectorize -h

usage: vectorize [-h] -o OUTPUT_BASE_PATH -t {openai_api,bert_cls,sbert} -m MODEL_PATH [--vec_dtype VEC_DTYPE] [--safetensors_dtype SAFETENSORS_DTYPE] [-r] [--batch_size BATCH_SIZE] [--vec_dim VEC_DIM] [--device DEVICE] [--max_retry MAX_RETRY] [--max_length MAX_LENGTH] [--hidden_layer HIDDEN_LAYER] [--normalize] [-v]

Vectorize text lines from stdin.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_BASE_PATH, --output_base_path OUTPUT_BASE_PATH
  -t {openai_api,bert_cls,sbert}, --vectorizer_type {openai_api,bert_cls,sbert}
  -m MODEL_PATH, --model_path MODEL_PATH
  --vec_dtype VEC_DTYPE
  --safetensors_dtype SAFETENSORS_DTYPE
  -r, --remove_vec_file
  --batch_size BATCH_SIZE
  --vec_dim VEC_DIM
  --device DEVICE
  --max_retry MAX_RETRY
  --max_length MAX_LENGTH
  --hidden_layer HIDDEN_LAYER
  --normalize
  -v, --verbose
  ```

#### Embedding Examples

##### Embed text by OpenAI's `text-embedding-ada-002`

Install `openai` package`:
```console
$ pip install openai
```

Then set API key and run `vectorize`` command:
```console
$ export HISTCONTROL=ignorespace  # do not save blankspace-started commands to history
$  export OPENAI_API_KEY=xxxx    # get secret key from https://platform.openai.com/account/api-keys
$ vectorize -t openai_api -m text-embedding-ada-002 -o ada-002 < input.txt

2023-08-29 07:28:44,514 INFO:__main__: Will create following files:
2023-08-29 07:28:44,514 INFO:__main__:   ada-002.vec
2023-08-29 07:28:44,514 INFO:__main__:   ada-002.vec.info
2023-08-29 07:28:44,514 INFO:__main__:   ada-002.safetensors
2023-08-29 07:28:44,514 INFO:__main__: embedding started
1000it [00:03, 293.99it/s]
2023-08-29 07:28:48,702 INFO:__main__: {
 "vec_dim": 1536,
 "vec_count": 1000,
 "vec_dtype": "float32",
 "vectorizer_type": "openai_api",
 "model_path": "text-embedding-ada-002"
}
2023-08-29 07:28:48,702 INFO:__main__: embedding finished
2023-08-29 07:28:48,702 INFO:__main__: convert to safetensors
2023-08-29 07:28:48,718 INFO:__main__: convert finished
2023-08-29 07:28:48,719 INFO:__main__: ada-002.vec removed
```

For other dtypes, add a option `--safetensors_dtype` to `vectorize`.
```console
$ vectorize -t openai_api -m text-embedding-ada-002 -o ada-002 --safetensors_dtype float16 < input.txt
```

##### Embed text by `cl-tohoku/bert-japanese-base-v3`

You need to use GPUs to embed text by BERT-like transformer models.

Install `transformers` and the tokenizer packages required in `cl-tohoku/bert-japanese-base-v3`:
```console
$ pip install transformers fugashi unidic-lite
```

Then run `vectorize`` command:
```console
$ vectorize -t bert_cls -m cl-tohoku/bert-base-japanese-v3 -o bert-base-japanese-v3 < input.txt

Some weights of BertForSequenceClassification were not initialized from the model checkpoint at cl-tohoku/bert-base-japanese-v3 and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
2023-08-29 07:26:04,673 INFO:__main__: Will create following files:
2023-08-29 07:26:04,673 INFO:__main__:   bert-base-japanese-v3.vec
2023-08-29 07:26:04,673 INFO:__main__:   bert-base-japanese-v3.vec.info
2023-08-29 07:26:04,673 INFO:__main__:   bert-base-japanese-v3.safetensors
2023-08-29 07:26:04,673 INFO:__main__: embedding started
1000it [00:04, 240.00027it/s]
2023-08-29 07:26:11,736 INFO:__main__: {
 "vec_dim": 768,
 "vec_count": 1000,
 "vec_dtype": "float32",
 "vectorizer_type": "bert_cls",
 "model_path": "cl-tohoku/bert-base-japanese-v3"
}
2023-08-29 07:26:11,736 INFO:__main__: embedding finished
2023-08-29 07:26:11,739 INFO:__main__: convert to safetensors
2023-08-29 07:26:11,750 INFO:__main__: convert finished
2023-08-29 07:26:11,751 INFO:__main__: bert-base-japanese-v3.vec removed
```

##### Embed text by `sentence-transformers` model

Install `sentence-transformers` package:
```console
$ pip install transformers sentence-transformers
```

Then run `vectorize`` command:
```console
$ vectorize -t sbert -m path_to_sbert_model -o sbert < input.txt

2023-08-29 07:26:53,544 INFO:__main__: Will create following files:
2023-08-29 07:26:53,544 INFO:__main__:   sbert.vec
2023-08-29 07:26:53,544 INFO:__main__:   sbert.vec.info
2023-08-29 07:26:53,544 INFO:__main__:   sbert.safetensors
2023-08-29 07:26:53,544 INFO:__main__: embedding started
1000it [00:02, 342.23it/s]
2023-08-29 07:26:56,757 INFO:__main__: {
 "vec_dim": 768,
 "vec_count": 1000,
 "vec_dtype": "float32",
 "vectorizer_type": "sbert",
 "model_path": "hysb_poor_mans_finetuned_posi/"
}
2023-08-29 07:26:56,757 INFO:__main__: embedding finished
2023-08-29 07:26:56,757 INFO:__main__: convert to safetensors
2023-08-29 07:26:56,768 INFO:__main__: convert finished
2023-08-29 07:26:56,769 INFO:__main__: sbert.vec removed
```

## Benchmarks

### Conditions and Environments

- GCP us-central1-b
  - balanced persistent disk 100GB
- ubuntu 22.04
  - cuda 11.8 (for GPUs)
- python 3.10.12
  - torch 2.0.1
- vectors
  - 768 dimension x 13,046,560 records = 10,019,758,080 elements
  - bfloat16 or float16 - 20.04[GB]
  - float32 - 40.08[GB]

### Results

<table>
<tr>
 <th rowspan=2> </th>
 <th rowspan=2>GCP Instance</th>
 <th rowspan=2>RAM</th>
 <th rowspan=2>Cost / Month</th>
 <th colspan=3>bfloat16/float16 in [sec]</th>
 <th colspan=3>float32 in [sec]</th>
</tr><tr>
 <th>score()</th><th>search()</th><th>search() targets</th>
 <th>score()</th><th>search()</th><th>search() targets</th>
</tr><tr>
 <th colspan=11 align="left">L4 GPU x 1</th>
</tr><tr>
 <td> </td><td>g2-standard-8</td><td>24GB (CUDA)</td><td>$633</td>
 <td>2.7e-4</td><td>5.3e-4</td><td>0.695</td>
 <td>-</td><td>-</td><td>-</td>
</tr><tr>
 <th colspan=11 align="left">A100 GPU x 1</th>
</tr><tr>
 <td> </td><td>a2-highgpu-1g</td><td>40GB (CUDA)</td><td>$2,692</td>
 <td>2.6e-4</td><td>5.6e-4</td><td>0.696</td>
 <td>2.5e-4</td><td>6.0e-4</td><td>0.697</td>
</tr><tr>
 <th colspan=11 align="left">Apple M1 Max 64GB</th>
</tr><tr>
 <td> </td><td>apple-m1-max</td><td>64GB (MPS)</td><td>-</td>
 <td>8.8e-4</td><td>1.3e-3</td><td>0.263</td>
 <td>1.6e-3</td><td>2.0e-3</td><td>0.274</td>
</tr><tr>
 <th colspan=11 align="left">Sapphire Rapids (SR)</th>
</tr><tr>
 <td>#1</td><td>c3-highmem-4</td><td>32GB</td><td>$216</td>
 <td>1.072</td><td>1.802</td><td>1.994</td>
 <td>-</td><td>-</td><td>-</td>
</tr><tr>
 <td>#2</td><td>c3-standard-8</td><td>32GB</td><td>$315</td>
 <td>0.533</td><td>1.217</td><td>1.413</td>
 <td>-</td><td>-</td><td>-</td>
</tr><tr>
 <td>#3</td><td>c3-highmem-8</td><td>64GB</td><td>$421</td>
 <td>0.531</td><td>1.209</td><td>1.398</td>
 <td>0.852</td><td>2.386</td><td>2.117</td>
</tr><tr>
 <td>#4</td><td>c3-highcpu-22</td><td>44GB</td><td>$702</td>
 <td>0.231</td><td>0.887</td><td>1.077</td>
 <td>0.392</td><td>1.948</td><td>1.695</td>
</tr><tr>
 <td>#5</td><td>c3-highcpu-44</td><td>88GB</td><td>$1,394</td>
 <td>0.174</td><td>0.829</td><td>1.033</td>
 <td>0.356</td><td>1.900</td><td>1.644</td>
</tr><tr>
 <th colspan=11 align="left">Cooper Lake (CL)</th>
</tr><tr>
 <td>#1</td><td>n2-highmem-4</td><td>32GB</td><td>$163</td>
 <td>1.250</td><td>2.029</td><td>2.217</td>
 <td>-</td><td>-</td><td>-</td>
</tr><tr>
 <td>#2</td><td>n2-standard-8</td><td>32GB</td><td>$237</td>
 <td>0.643</td><td>1.388</td><td>1.671</td>
 <td>-</td><td>-</td><td>-</td>
</tr><tr>
 <td>#3</td><td>n2-highcpu-32</td><td>32GB</td><td>$702</td>
 <td>0.259</td><td>0.969</td><td>1.196</td>
 <td>-</td><td>-</td><td>-</td>
</tr><tr>
 <td>#4</td><td>n2-highmem-8</td><td>64GB</td><td>$316</td>
 <td>0.686</td><td>1.422</td><td>1.628</td>
 <td>0.923</td><td>2.410</td><td>2.255</td>
</tr><tr>
 <td>#5</td><td>n2-standard-16/td><td>64GB</td><td>$464</td>
 <td>0.375</td><td>1.084</td><td>1.307</td>
 <td>0.508</td><td>1.967</td><td>1.820</td>
</tr><tr>
 <td>#6</td><td>n2-highcpu-48</td><td>48GB</td><td>$1,015</td>
 <td>0.209</td><td>0.916</td><td>1.161</td>
 <td>0.370</td><td>1.878</td><td>1.743</td>
</tr><tr>
 <th colspan=11 align="left">Haswell (HW)</th>
</tr><tr>
 <td>#1</td><td>n1-highmem-8</td><td>52GB</td><td>$251</td>
 <td>62.317</td><td>63.095</td><td>63.461</td>
 <td>0.876</td><td>2.760</td><td>2.727</td>
</tr><tr>
 <td>#2</td><td>n1-standard-16</td><td>60GB</td><td>$398</td>
 <td>62.218</td><td>63.048</td><td>63.397</td>
 <td>0.530</td><td>2.365</td><td>2.298</td>
</tr><tr>
 <td>#3</td><td>n1-highcpu-64</td><td>57GB</td><td>$1,169</td>
 <td>62.141</td><td>63.026</td><td>63.818</td>
 <td>0.530</td><td>2.325</td><td>2.280</td>
</tr><tr>
</table>
