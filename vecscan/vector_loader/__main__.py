import argparse
import logging
import sys

from . import VectorLoader
from ..utils import ARCHITECTURE_DEFAULT_DTYPE


logger = logging.getLogger("vecscan")


def parse_args():
    parser = argparse.ArgumentParser(description="A tool converting vector data to a safetensors file")
    parser.add_argument("-f", "--input_format", choices=["csv", "jsonl", "binary"], required=True)
    parser.add_argument("-o", "--output_safetensors_path", type=str, required=True)
    parser.add_argument("-d", "--vec_dim", type=int, default=None)
    parser.add_argument("-n", "--normalize", action="store_true", default=False)
    parser.add_argument("-s", "--skip_first_line", action="store_true", default=False)
    parser.add_argument("-t", "--target_field", type=str, default=None)
    parser.add_argument("--input_dtype", type=str, default="float32")
    parser.add_argument("--safetensors_dtype", type=str, default=ARCHITECTURE_DEFAULT_DTYPE)
    parser.add_argument("--shard_size", type=int, default=2**32)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
    args = parse_args()
    logger.setLevel(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.debug(args)
    logger.info(f"convert sys.stdin to {args.output_safetensors_path}")
    logger.info(f"  input_format={args.input_format}, input_dtype={args.input_dtype}, vec_dim={args.vec_dim}, target_field={args.target_field}, normalize={args.normalize}, safetensors_dtype={args.safetensors_dtype}, shard_size={args.shard_size}")
    vector_loader = VectorLoader.create(
        input_format=args.input_format,
        input_dtype=args.input_dtype,
        vec_dim=args.vec_dim,
        skip_first_line=args.skip_first_line,
        target_field=args.target_field,
        normalize=args.normalize,
        safetensors_dtype=args.safetensors_dtype,
        shard_size=args.shard_size,
    )
    scanner = vector_loader.create_vector_scanner(sys.stdin)
    scanner.save_file(args.output_safetensors_path)
    logger.info(f"{len(scanner)} records converted")


if __name__ == "__main__":
    main()
