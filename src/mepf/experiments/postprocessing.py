import json
import logging
import os

import numpy as np
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
    """
    Load all the result from the grid experiment
    """
    logging.info(f"Loading results at {save_dir}")
    res = []
    for method in grid["method"]:
        for problem in grid["problem"]:
            sub_dir = save_dir / method / problem
            if not sub_dir.exists():
                continue
            for filename in os.listdir(sub_dir):
                file_path = sub_dir / filename
                with open(file_path, "rt") as f:
                    res += [json.loads(line) for line in f]
    all_runs = pd.DataFrame(res)
    return all_runs


def scatter_result(save_dir, res_dir, grid):
    """
    We create more file to simplify future computation
    """
    logging.info("Scattering results")
    local_grid = {}
    for method in grid["method"]:
        local_grid["method"] = [method]
        for problem in grid["problem"]:
            local_grid["problem"] = [problem]
            res = load_all_results(save_dir, local_grid)
            if res.empty:
                continue
            # res.dropna(inplace=True)
            for m in grid["num_classes"]:
                for delta in grid["delta"]:
                    ind = res["m"] == m
                    ind &= res["delta"] == delta
                    tmp = res[ind]
                    tmp = tmp[
                        [
                            "nb_queries",
                            "success",
                            "n_data",
                            "n_sanov",
                            "n_elim",
                            "constant",
                            "seed",
                        ]
                    ].reset_index(drop=True)

                    outdir = (
                        res_dir / method / problem / str(m) / str(int(-np.log2(delta)))
                    )
                    outdir.mkdir(parents=True, exist_ok=True)
                    tmp.to_pickle(outdir / "res.pkl")
            logging.info(f"... done for {method} {problem}")


def process_results(res_dir, grid, func):
    """
    Easy result processing thanks to the previous scattering
    """
    all_res = []
    for method in grid["method"]:
        for problem in grid["problem"]:
            for m in grid["num_classes"]:
                for delta in grid["delta"]:
                    sub_dir = (
                        res_dir / method / problem / str(m) / str(int(-np.log2(delta)))
                    )
                    if not sub_dir.exists():
                        logging.warning(f"Missing {sub_dir}")
                    file_path = sub_dir / "res.pkl"
                    try:
                        res = pd.read_pickle(file_path)
                    except FileNotFoundError as e:
                        logger.warning(e)
                        continue
                    for line in func(res, method, problem, m, delta):
                        all_res.append(line)
    return all_res


def get_validity(res_dir, grid, tol=0, verbose=False):
    """
    Get experiments where all methods work decently
    """

    def valid_report(res, method, problem, m, delta):
        if res is None:
            yield [method, problem, m, delta, [], False]
            return

        valid_constants = []
        for constant in grid["constant"]:
            if method in ["ES", "AS", "TS", "HTS"] and constant != 1:
                continue

            tmp = res[res["constant"] == constant]

            if len(tmp) < 1 / delta:
                error = 1
            elif method in ["ES", "AS", "TS", "HTS"]:
                error = 1 - tmp["success"].mean()
            else:
                error = 1 - (tmp["success"] & (tmp["nb_queries"] != -1)).mean()

            valid = error <= (1 + tol) * delta
            if valid:
                valid_constants.append(constant)
        yield [method, problem, m, delta, valid_constants, bool(valid_constants)]

    is_valid = process_results(res_dir, grid, valid_report)
    validity = pd.DataFrame(
        is_valid, columns=["method", "problem", "m", "delta", "constants", "valid"]
    )

    if verbose:
        tmp = validity.groupby(["problem", "m", "delta"])["valid"].mean(())
        valid_list = list(tmp[tmp == 1].index.values)

        if valid_list:
            setup = valid_list[0]
            ind = (
                (validity["problem"] == setup[0])
                & (validity["m"] == setup[1])
                & (validity["delta"] == setup[2])
            )
            for setup in valid_list[1:]:
                ind |= (
                    (validity["problem"] == setup[0])
                    & (validity["m"] == setup[1])
                    & (validity["delta"] == setup[2])
                )
            valid_df = validity[ind]
        return validity, valid_list, valid_df

    return validity


def get_statistics(res_dir, grid, tol=0):
    def stat_report(res, method, problem, m, delta):
        # number of experiments and of required success
        num_exp = int(np.ceil(10 / delta))
        # num_best = int(num_exp * (1 - delta * (1 + tol)))
        num_best = num_exp

        # remove experiments that were not launched & clean data
        res["nb_queries"] = res["nb_queries"].astype(float)
        res = res.dropna()
        # # run stopped by us may be considered unsuccessful
        # res.loc[res["n_elim"] == res["n_data"], "success"] = False

        # # unsuccessful runs will not count
        # res.loc[res["success"] == False, "nb_queries"] = np.inf

        for constant in grid["constant"]:
            if method in ["ES", "AS", "TS", "HTS"] and constant != 1:
                continue
            tmp = res[res["constant"] == constant]

            delta_effective = 1 - tmp["success"].mean()
            ratio_exp = len(tmp) / num_exp
            # num_best = int(len(tmp) * (1 - delta * (1 + tol)))
            best = tmp.sort_values("nb_queries")["nb_queries"][:num_best]

            if np.isinf(best.values).any():
                best_mean = np.nan
                best_std = np.nan
            else:
                best_mean = best.mean()
                best_std = best.std()
            yield [
                method,
                problem,
                constant,
                m,
                delta,
                delta_effective,
                best_mean,
                best_std,
                ratio_exp,
            ]

    out = process_results(res_dir, grid, stat_report)
    out = pd.DataFrame(
        out,
        columns=[
            "method",
            "problem",
            "constant",
            "m",
            "delta",
            "delta_effective",
            "best_mean",
            "best_std",
            "ratio_exp",
        ],
    )
    return out


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Mode estimation with partial feeback results pre-processing",
    )
    parser.add_argument(
        "--res-dir",
        type=str,
        default="results/",
        help="postprocessed saving directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/",
        help="preprocessed result directory",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0,
        help="tolerance on confidence guarantee, delta <- delta(1 + tol)",
    )
    config = parser.parse_args()
    save_dir = Path(config.save_dir)
    res_dir = Path(config.res_dir)
    tol = config.tol

    grid = {
        "method": ["ES", "AS", "TS", "E", "SE"],
        "problem": ["one", "two", "geometric"],
        "num_classes": [30, 100, 300, 1000],
        "delta": [2**-i for i in range(1, 10)],
        "constant": [0.1, 0.3, 1, 3, 10, 24],
    }

    scatter_result(save_dir, res_dir, grid)
    validity, valid_list, valid_df = get_validity(res_dir, grid, tol=tol, verbose=True)
    logging.info(f"number of valid experiments: {len(valid_list)}")
    stat_report = get_statistics(res_dir, grid, tol=tol)
    stat_report.to_csv(res_dir / "stat.csv")
    logging.info(f"result saved at {res_dir / 'stat.csv'}")
