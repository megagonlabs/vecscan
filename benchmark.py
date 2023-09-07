import argparse
import json
import logging
import random
from statistics import mean
from time import perf_counter

import torch

from vecscan import VectorScanner


logger = logging.getLogger("vecscan")


def parse_args():
    parser = argparse.ArgumentParser(description="A benchmark tool for vecscan")
    parser.add_argument("safetensors_paths", nargs="+")
    parser.add_argument("-o", "--output_json_path", type=str, default="benchmark.json")
    parser.add_argument("-d", "--device_csv", type=str, default="cpu,cuda:0")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def benchmark_file(safetensors_path: str, device: str) -> dict:
    prev = perf_counter()
    scanner = VectorScanner.load_file(safetensors_path, device=device, break_in=False)
    stats = {
        "vectors.shape": str(scanner.shape),
        "vectors.dtype": str(scanner.dtype),
    }
    now = perf_counter()
    stats["load_file(break_in=False)"] = now - prev
    prev = now
    query_vec = scanner[0]
    scanner.search(query_vec)
    now = perf_counter()
    stats["break-in scan"] = now - prev
    target_ids = list(range(0, len(scanner), 2))
    for title, f in [
        ("score()", lambda q: scanner.score(q)),
        ("search()", lambda q: scanner.search(q)),
        ("search(target_ids)", lambda q: scanner.search(q, target_ids=target_ids)),
    ]:
        trials = []
        for _ in range(5):
            query_vec = scanner[random.randint(0, len(scanner) - 1)]
            f(query_vec)
            now = perf_counter()
            trials.append(now - prev)
            prev = now
        mean_search_time = mean(sorted(trials)[1:4])
        stats[title] = mean_search_time
    return stats


def main():
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
    args = parse_args()
    logger.setLevel(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.debug(args)
    results = {}
    for safetensors_path in args.safetensors_paths:
        results[safetensors_path] = {}
        for device in args.device_csv.split(","):
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.info("torch.cuda.is_available() == False, skipping cuda")
                continue
            result = benchmark_file(safetensors_path, device)
            results[safetensors_path][device] = result
            logger.info(f"{safetensors_path=}, {device=}, {result=}")
    with open(args.output_json_path, "w", encoding="utf8") as fout:
        json.dump(results, fout, indent=1, ensure_ascii=False)
        print(file=fout)
    logger.info(f"Benchmark results were dumped to {args.output_json_path}.")


if __name__ == "__main__":
    main()
