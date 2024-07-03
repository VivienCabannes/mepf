import json

import numpy as np

from mepf import (
    BatchElimination,
    Elimination,
    ExhaustiveSearch,
    RoundFreeSetElimination,
    RoundFreeTruncatedSearch,
    TruncatedSearch,
    nb_data_required,
    nb_elim_data_required,
)
from mepf.data import geometric, one_vs_all, sample_dirichlet, two_vs_all

DATA_MAX = 10_000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_problem(m: int, problem: str, delta: float, rng: np.random.Generator):
    match problem:
        case "dirichlet":
            alpha = np.zeros(m) + 0.1
            proba = sample_dirichlet(alpha, generator=rng)
        case "one":
            proba = one_vs_all(m, p1=1 / 2)
        case "two":
            proba = two_vs_all(m, p1=0.2, diff=0.14)
        case "geometric":
            proba = geometric(m)
    proba = rng.permutation(proba)
    n_sanov = nb_data_required(proba, delta)
    n_elim = nb_elim_data_required(proba, delta)
    if n_elim >= DATA_MAX:
        return None, n_sanov, n_elim, None
    y_cat = rng.choice(m, size=n_elim, p=proba)
    return y_cat, n_sanov, n_elim, proba


def experiments(config, seed):
    rng = np.random.default_rng(seed)
    y_cat, n_sanov, n_elim, proba = generate_problem(
        config.num_classes, config.problem, config.delta, rng
    )
    if n_elim >= DATA_MAX:
        return {
            "n_sanov": n_sanov,
            "n_elim": n_elim,
            "problem": config.problem,
            "m": config.num_classes,
            "delta": config.delta,
        }
    m = len(proba)

    match config.method:
        case "ES":
            y_cat = y_cat[:n_sanov]
            model = ExhaustiveSearch(m, adaptive=False)
            for y in y_cat:
                model(y)
            # cla, fre = np.unique(y_cat, return_counts=True)
            # assert fre[model.mode.label == cla] == np.max(fre)
        case "AS":
            y_cat = y_cat[:n_sanov]
            model = ExhaustiveSearch(m, adaptive=True)
            for y in y_cat:
                model(y)
            # cla, fre = np.unique(y_cat, return_counts=True)
            # assert fre[model.mode.label == cla] == np.max(fre)
        case "TS":
            y_cat = y_cat[:n_sanov]
            model = TruncatedSearch(m)
            model(y_cat)
            model = model.back_end
            # cla, fre = np.unique(y_cat, return_counts=True)
            # assert fre[model.mode.label == cla] == np.max(fre)
        case "HTS":
            y_cat = y_cat[:n_sanov]
            model = RoundFreeTruncatedSearch(m)
            for y in y_cat:
                model(y)
            # cla, fre = np.unique(y_cat, return_counts=True)
            # assert fre[model.mode.label == cla] == np.max(fre)
        case "E":
            model = Elimination(
                m, confidence_level=1 - config.delta, constant=config.constant
            )
            for y in y_cat:
                model(y)
                if model.eliminated.sum() == m - 1:
                    break
        case "SE":
            model = BatchElimination(
                m, confidence_level=1 - config.delta, constant=config.constant
            )
            r = 0
            s_ind = 0
            while s_ind < len(y_cat):
                r += 1
                epsilon = (2 / 3) ** r / (4 * m)
                bsz = 2**r
                e_ind = s_ind + bsz
                model(y_cat[s_ind:e_ind], epsilon)
                s_ind = e_ind
                if model.eliminated.sum() == m - 1:
                    break
        case "HSE":
            model = RoundFreeSetElimination(
                m, confidence_level=1 - config.delta, constant=config.constant
            )
            for y in y_cat:
                model(y)
                if model.eliminated.sum() == m - 1:
                    break

    results = {
        "nb_queries": model.nb_queries,
        "success": bool(model.mode.label == np.argmax(proba)),
        "n_data": model.root.value,
        "n_sanov": n_sanov,
        "n_elim": n_elim,
        "method": config.method,
        "problem": config.problem,
        "m": config.num_classes,
        "delta": config.delta,
        "constant": config.constant,
        "seed": seed,
    }
    return results


if __name__ == "__main__":
    import argparse
    import logging
    import sys
    from itertools import product
    from pathlib import Path

    # Logging
    logger = logging.getLogger("grid")
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

    parser = argparse.ArgumentParser(
        description="bandit ECB",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "ES",
            "AS",
            "TS",
            "HTS",
            "E",
            "SE",
            "HSE",
        ],
        default="AS",
        help="algorithm to estimate the mode from partial feedback",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="number of classes",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=[
            "dirichlet",
            "one",
            "two",
            "geometric",
        ],
        default="dirichlet",
        help="problem to solve",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="tolerated percentage of error",
    )
    parser.add_argument(
        "--constant",
        type=float,
        default=24,
        help="constant to resize confidence intervals",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="interactive mode (in constrast to grid run)",
    )
    parser.add_argument(
        "--num-tasks",
        default=1,
        type=int,
        help="number of tasks to split the grid runs into",
    )
    parser.add_argument(
        "--task-id",
        default=1,
        type=int,
        help="task id, from 1 to num_tasks",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/",
        help="saving directory",
    )
    config = parser.parse_args()

    # Interactive mode
    if config.interactive:
        res = experiments(config, seed=0)
        print("Results:")
        for k, v in res.items():
            print("\t", k, v)
        sys.exit(0)

    grid = {
        "method": ["ES", "AS", "TS", "HTS", "E", "SE", "HSE"],
        "problem": ["dirichlet", "one", "two", "geometric"],
        "num_classes": [15, 30, 100, 300, 1000, 3000],
        "delta": [2**-i for i in range(1, 10)],
        "constant": [0.1, 0.3, 1, 3, 10, 24],
    }

    logger.info(
        f"Number of experiments: {len(list(product(*grid.values()))) // config.num_tasks}"
    )

    # Run grids
    for i, vals in enumerate(product(*grid.values())):
        # Splitting the grid into tasks
        if i % config.num_tasks != (config.task_id - 1):
            continue

        # Setting configuration
        for k, v in zip(grid.keys(), vals):
            setattr(config, k, v)
        logger.info(f"Config: {config}")

        if config.method in ["ES", "AS", "TS", "HTS"] and config.constant != 1:
            continue

        # Output file
        outdir = Path(config.save_dir) / config.method / config.problem
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"worker_{config.task_id}.jsonl"

        # Running experiment
        num_exp = int(np.ceil(10 / config.delta))
        for seed in range(num_exp):
            try:
                res = experiments(config, seed)
            except Exception as e:
                logger.warning(f"Error for configuration: {config}")
                logger.warning(e)
                continue

            # Saving results
            with open(outfile, "a") as f:
                print(json.dumps(res, cls=NpEncoder), file=f)
