import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

WIDTH = 7  # inches (from ICML style file)
HEIGHT = 7 / 1.618  # golden ratio

rc("font", family="serif", size=8)
usetex = not subprocess.run(["which", "pdflatex"]).returncode
rc("text", usetex=usetex)
if usetex:
    rc("text.latex", preamble=r"\usepackage{times,amsmath,amsfonts,amssymb}")

res = pd.read_csv("~/Code/mepf/results/stat.csv")
res["okay"] = res["delta"] >= res["delta_effective"]
res = res.dropna()

methods = ["AS", "ES", "TS", "E", "SE"]
constants = [0.1, 0.3, 1, 3, 10, 24]
ms = [15, 30, 100, 300, 1000, 3000]

for problem in ["geometric", "two", "one"]:
    if problem == "geometric":
        ms = [100]
        methods = ["ES", "AS", "TS"]
    elif problem == "two":
        ms = [300]
        methods = ["E", "TS"]
    elif problem == "one":
        ms = [1000]
        methods = ["E", "SE", "TS"]
    for i, m in enumerate(ms):
        ind = res["problem"] == problem
        ind &= res["m"] == m

        leg = []
        # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(.8 * WIDTH, .4 * HEIGHT))
        fig = plt.figure(figsize=(0.8 * WIDTH, 0.4 * HEIGHT))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35)
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        for method in methods:
            met_ind = ind & (res["method"] == method)
            if method == "ES":
                color = "black"
            elif method == "E":
                color = "C0"
            elif method == "SE":
                color = "C1"
            elif method == "HSE":
                color = "C4"
            elif method == "AS":
                color = "C2"
            elif method == "TS":
                color = "C3"
            elif method == "HTS":
                color = "C5"
            for c in constants:
                if method not in ["E", "SE", "HSE"] and c != 1:
                    continue
                if method == "SE" and c != 0.1:
                    continue
                if method == "E" and c != 0.3:
                    continue
                tmp = res[met_ind & (res["constant"] == c)]

                start = 0
                (a,) = ax.plot(
                    np.arange(len(tmp))[start:] + 1,
                    tmp["best_mean"].values[start:],
                    label=f"{method}",
                    color=color,
                )
                ax.fill_between(
                    np.arange(len(tmp))[start:] + 1,
                    tmp["best_mean"].values[start:]
                    - 0.5 * tmp["best_std"].values[start:],
                    tmp["best_mean"].values[start:]
                    + 0.5 * tmp["best_std"].values[start:],
                    alpha=0.1,
                    color=color,
                )
                # a, = ax.plot(tmp['delta_effective'], tmp['delta'], alpha=min(1, 5*c), color=color)
                leg.append(a)

                ax2.plot(
                    np.arange(len(tmp))[start:] + 1,
                    tmp["delta_effective"][start:],
                    label=f"{method}",
                    color=color,
                )

        # ax.legend(leg, ['ES-AS-TS', 'E, c=0.1', 'E, c=0.3', 'SE, c=0.1', 'SE, c=0.3'], fontsize=6)
        ax.legend(fontsize=6)
        ax.set_xlabel(r"$\log_2(1/\delta)$", fontsize=10)
        ax.set_ylabel(r"$\mathbb{E}[T_\delta]$", fontsize=10)
        ax2.set_xlabel(r"$\log_2(1/\delta)$", fontsize=10)
        ax2.set_ylabel(r"$\delta_{\text{eff}}$", fontsize=10)
        fig.suptitle(
            {
                "one": "One good arm",
                "two": "Two good arms",
                "dirichlet": "Direchlet arms",
                "geometric": "Geometric arms",
            }[problem]
            + f", {m=}",
            fontsize=10,
        )
        # fig.savefig(f'{problem}_{m}.pdf', bbox_inches='tight', pad_inches=0)
