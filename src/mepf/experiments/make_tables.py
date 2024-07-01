
from pathlib import Path

import pandas as pd

# save_dir = Path.home() / 'bandit_exp'
save_dir = Path.home()
STR = ''

grid = {
    # "method": ["SE", "UUCB", "MUCB", "LUCB", "bDCB", "bMMM", "bEBTC", "cECB", "cDCB", "cMMM", "cEBTC"],
    "method": ["MUCB", "bEBTC", "cECB", "cEBTC"],
    # "arms": [3, 10, 30, 100],
    "arms": [3, 30],
    # "budget": [1000, 10000, 30000],
    "budget": [30000],
    "p_lim": [.5, 1], 
    "delta": [.01, .05, .1, .5], 
}
confidence = grid['method'][-2:]
budget = grid['method'][:-2]
# grid = {
#     "method": ["SE", "UUCB", "MUCB", "LUCB", "bDCB", "bMMM", "bEBTC", "cECB", "cDCB", "cMMM", "cEBTC"],
#     "arms": [3, 10, 30, 100],
#     "budget": [1000, 10000, 30000],
#     "p_lim": [.5, 1],
#     "delta": [.01, .05, .1, .5],
# }
# confidence = grid['method'][-4:]
# budget = grid['method'][:-4]

term_res = []
online_res = []
for method in grid['method']:
    if method[0] == 'c':
        term_res.append(pd.read_pickle(save_dir / f'{method}.pkl'))
    else:
        online_res.append(pd.read_pickle(save_dir / f'{method}.pkl'))

term_res = pd.concat(term_res, axis=0)
online_res = pd.concat(online_res, axis=0)

online_res['success'] = online_res['pulls'] != -1
ind = online_res['success'] == True
groups = ['method', 'delta', 'budget', 'p_lim', 'arms']
success = online_res.groupby(groups)['success'].mean()
pulls_mean = online_res[ind].groupby(groups)['pulls'].mean().astype(int)
pulls_std = online_res[ind].groupby(groups)['pulls'].std().astype(int)


all_tables = {}
for p_lim in grid['p_lim']:
    for arms in grid['arms']:
        # if arms not in [3, 30]:
        #     continue
        for time in grid['budget']:
            # if time != 30000:
            #     continue
            table = '\\begin{table}[t]\n  \\centering\n  \\begin{tabular}{|c|c|c|c|c|}\n    \\hline\n'
            table += '    methods '
            for delta in grid['delta']:
                table += f'& $\delta={delta}$ '
            table += '\\\\\n    \\hline\n'
            for method in online_res['method'].unique():
                table += f'    {method} '
                for delta in grid['delta']:
                    groups = (method, delta, time, p_lim, arms)
                    table += f'& {int(1000 * success[groups]) / 10} \% - {pulls_mean[groups]} $\pm$ {pulls_std[groups]} '
                table += '\\\\\n'
            table += '    \\hline\n    \\end{tabular}\n\\caption{Online experiments: $T =' + str(time) + '$, $p_{\\lim} = ' + str(p_lim) + '$, $m=' + str(arms) + '$}\n\\label{tab:term-' + f'{time}-{p_lim}-{arms}' + '}\n\\end{table}'
            all_tables[time, p_lim] = table
            STR += table + '\n\n'
            print(table, '\n\n')


term_res['success'] = term_res['pulls'] != -1
ind = term_res['success'] == True
groups = ['method', 'delta', 'budget', 'p_lim', 'arms']
success = term_res.groupby(groups)['success'].mean()
pulls_mean = term_res[ind].groupby(groups)['pulls'].mean().astype(int)
pulls_std = term_res[ind].groupby(groups)['pulls'].std().astype(int)

term_res['stop_success'] = term_res['ts'].apply(max) != term_res['budget']
term_res['stop_success'] &= term_res['pulls'] != -1
ind = term_res['stop_success'] == True
stop_success = term_res.groupby(groups)['stop_success'].mean()
stop_pulls_mean = term_res[ind].groupby(groups)['pulls'].mean().astype(int)
stop_pulls_std = term_res[ind].groupby(groups)['pulls'].std() // 1


all_tables = {}
for p_lim in grid['p_lim']:
    for arms in grid['arms']:
        # if arms not in [3, 30]:
        #     continue
        for time in grid['budget']:
            # if time != 30000:
            #     continue
            table = '\\begin{table}[t]\n  \\centering\n  \\begin{tabular}{|c|c|c|c|c|}\n    \\hline\n'
            table += '    methods '
            for delta in grid['delta']:
                table += f'& $\delta={delta}$ '
            table += '\\\\\n    \\hline\n'
            for method in term_res['method'].unique():
                if method == 'cMMM':
                    continue
                table += f'    {method} '
                for delta in grid['delta']:
                    groups = (method, delta, time, p_lim, arms)
                    try:
                        table += f'& {int(1000 * success[groups]) / 10} \% - {pulls_mean[groups]} $\pm$ {pulls_std[groups]:.0f} '
                    except KeyError:
                        table += f'& - '
                table += '\\\\\n        '
                for delta in grid['delta']:
                    groups = (method, delta, time, p_lim, arms)
                    try:
                        table += f'& {int(1000 * stop_success[groups]) / 10} \% - {stop_pulls_mean[groups]} $\pm$ {stop_pulls_std[groups]:.0f} '
                    except KeyError:
                        table += f'& - '
                table += '\\\\\n     \\hline\n'
            table += '    \\end{tabular}\n\\caption{Termination experiments: $T =' + str(time) + '$, $p_{\\lim} = ' + str(p_lim) + '$, $m=' + str(arms) + '$}\n\\label{tab:conf-' + f'{time}-{p_lim}-{arms}' + '}\n\\end{table}'
            all_tables[time, p_lim] = table
            STR += table + '\n\n'
            print(table, '\n\n')

with open('experiments.tex', 'w') as f:
    f.write(STR)
