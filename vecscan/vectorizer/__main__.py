import argparse
import json
import logging
import os
import sys

import torch

from . import Vectorizer, ARCHITECTURE_DEFAULT_DTYPE
from .. import convert_vec_to_safetensors


logger = logging.getLogger("vescan")


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorize text lines from stdin")
    parser.add_argument("-o", "--output_base_path", type=str, required=True)
    parser.add_argument("-t", "--vectorizer_type", choices=["openai_api", "bert_cls", "sbert"], required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--vec_dtype", type=str, default="float32")
    parser.add_argument("--safetensors_dtype", type=str, default=ARCHITECTURE_DEFAULT_DTYPE)
    parser.add_argument("-r", "--remove_vec_file", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--vec_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_retry", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--hidden_layer", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
    args = parse_args()
    logger.setLevel(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.debug(args)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    vectorizer = Vectorizer.create(**args_dict)

    output_vec_path = args.output_base_path + ".vec"
    output_info_path = output_vec_path + ".info"
    output_safetensors_path = args.output_base_path + ".safetensors"
    logger.info("Will create following files:")
    for _ in [output_vec_path, output_info_path, output_safetensors_path]:
        logger.info("  " + _)

    with open(output_vec_path, "wb") as fout:
        logger.info("embedding started")
        vec_count = vectorizer.vectorize_file(sys.stdin, fout)
    info = vectorizer._make_info(vec_count)
    with open(output_info_path, "w") as fout:
        json.dump(info, fout, indent=1)

    if vec_count > 0:
        logger.info(json.dumps(info, indent=1))
        logger.info("embedding finished")
    else:
        logger.warning("no input lines")
        return

    vec_dim = vectorizer.vec_dim
    vectorizer = None
    logger.info(f"convert to safetensors for {args.safetensors_dtype}: {output_safetensors_path}")
    convert_vec_to_safetensors(output_vec_path, vec_dim, args.vec_dtype, args.safetensors_dtype, output_safetensors_path)
    logger.info("convert finished")
    if args.remove_vec_file:
        os.remove(output_vec_path)
        logger.info(f"{output_vec_path} removed")
    else:
        logger.info(f"You can remove '{output_vec_path}' if you will not create other safetensors files with differenet dtypes")


if __name__ == '__main__':
    main()
