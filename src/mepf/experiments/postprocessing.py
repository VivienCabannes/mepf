
import json
import logging
import os

import pandas as pd
import numpy as np


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


def scatter_result(save_dir, res_dir, grid):
    """
    We create more file to simplify future computation
    """
    local_grid = {}
    for method in grid['method']:
        local_grid['method'] = [method]
        for problem in grid['problem']:
            local_grid['problem'] = [problem]
            res = load_all_results(save_dir, local_grid)
            # res.dropna(inplace=True)
            for m in grid['num_classes']:
                for delta in grid['delta']:
                    ind = res['m'] == m
                    ind &= res['delta'] == delta
                    tmp = res[ind]
                    tmp = tmp[['nb_queries', 'success', 'n_data', 'constant', 'seed']].reset_index(drop=True)

                    outdir = res_dir / method / problem / str(m) / str(int(-np.log2(delta)))
                    outdir.mkdir(parents=True, exist_ok=True)
                    tmp.to_pickle(outdir / 'res.pkl')


def load_scatter_results(res_dir, grid):
    res = []
    for method in grid['method']:
        for problem in grid['problem']:
            for m in grid['num_classes']:
                for delta in grid['delta']:
                    sub_dir = res_dir / method / problem / str(m) / str(int(-np.log2(delta)))
                    if not sub_dir.exists():
                        continue
                    file_path = sub_dir / 'res.pkl'
                    res.append(pd.read_pickle(file_path))
    if res:
        all_runs = pd.concat(res, ignore_index=True)
        return all_runs


def get_validity(res_dir, grid, tol=0.1):
    is_valid = []
    local_grid = {}
    for method in grid['method']:
        local_grid['method'] = [method]
        for problem in grid['problem']:
            local_grid['problem'] = [problem]
            for m in grid['num_classes']:
                local_grid['num_classes'] = [m]
                if m != 10:
                    continue
                for delta in grid['delta']:
                    local_grid['delta'] = [delta]

                    res = load_scatter_results(res_dir, local_grid)

                    valid_constants = []
                    for constant in grid['constant']:
                        if method in ['ES', 'AS', 'TS', 'HTS'] and constant != 1:
                            continue

                        tmp = res[res['constant'] == constant]

                        if len(tmp) < 1 / delta:
                            error = 1
                        elif method in ['ES', 'AS', 'TS', 'HTS']:
                            error = 1 - tmp['success'].mean()
                        else:
                            error = 1 - (tmp['success'] & (tmp['nb_queries'] != -1)).mean()

                        valid = error <= (1 + tol) * delta
                        if valid:
                            valid_constants.append(constant)
                    is_valid.append([method, problem, m, delta, valid_constants, bool(valid_constants)])

    valid_df = pd.DataFrame(is_valid, columns=['method', 'problem', 'm', 'delta', 'constants', 'valid'])
    return valid_df


def get_valid_setup(valid_df):
    tmp = valid_df.groupby(['problem', 'm', 'delta'])['valid'].mean(())
    valid_setup = list(tmp[tmp == 1].index.values)

    if valid_setup:
        setup = valid_setup[0]
        ind = (valid_df['problem'] == setup[0]) & (valid_df['m'] == setup[1]) & (valid_df['delta'] == setup[2])

        for setup in valid_setup[1:]:
            ind |= (valid_df['problem'] == setup[0]) & (valid_df['m'] == setup[1]) & (valid_df['delta'] == setup[2])
        return valid_df[ind]


def load_valid_exp(res_dir, valid_setup):
    res = []
    for line in valid_setup.values:
        method, problem, m, delta, constants = line[:5]
        local_grid = {
            'method': [method],
            'problem': [problem],
            'num_classes': [m],
            'delta': [delta],
        }
        cur = load_scatter_results(res_dir, local_grid)
        cur = cur[cur['constant'].isin(constants)]
        cur['problem'] = problem
        cur['m'] = m
        cur['method'] = method
        cur['delta'] = delta
        res.append(cur)
    res = pd.concat(res)
    return res


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
