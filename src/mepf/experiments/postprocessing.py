
import json
import logging
import os

import pandas as pd


logger = logging.getLogger("preprocess")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="{asctime} {levelname} [{filename}:{lineno}] {message}",
    style="{",
    datefmt="%H:%M:%S",
    level="INFO",
    handlers=[
        logging.StreamHandler(),
    ],
)


def load_all_results(save_dir, grid):
    res = []
    for method in grid['method']:
        for problem in grid['problem']:
            sub_dir = save_dir / method / problem
            if not sub_dir.exists():
                continue
            for filename in os.listdir(sub_dir):
                file_path = sub_dir / filename
                with open(file_path, 'rt') as f:
                    res += [json.loads(line) for line in f]
    all_runs = pd.DataFrame(res)
    return all_runs


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="bandit results pre-processing",
    )
    parser.add_argument(
        "--dir-path",
        type=str,
        default=".",
        help="postprocessed saving directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/checkpoint/vivc/mepf",
        help="preprocessed result directory",
    )
    config = parser.parse_args()

    grid = {
        "method": ["ES", "AS", "TS", "HTS", "E", "SE", "HSE"],
        "problem": ["one", "two", "geometric"],
        "num_class": [3, 10, 30, 100, 300, 1000],
        "delta": [2 ** -i for i in range(1, 10)],
        "constant": [0.5, 1, 3, 10, 24],
    }

    res = load_all_results(Path(config.save_dir), grid)
