# Expected structure:
# root
# ├── {network_id}
# │   ├── {resolution}
# │   │   ├── {replicate_id}
# │   │   │   └── {COMP_FN}

# Expected format of {COMP_FN}:
# stat,stat_type,distance_type,distance

# Output format:
# {output_dir}/tables/{network_id}_{resolution}.csv
# stat,stat_type,distance_type,distance_count_{name},distance_mean_{name},distance_std_{name} for name in names
# for every (network, clustering) pair where there are at least two simulators producing at least one replicate each

# {output_dir}/all_successes.csv
# Network,Resolution,{name} for name in names
# for every (network, clustering) pair

import argparse
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Q = 1.0

COMP_FNS = [
    'compare_output.csv',
    'compare_stats.csv',
]

MINMAX_BOUNDED_SCALARS = {
    # 'deg_assort': (-0.2, 0.2),
    # 'local_ccoeff': (-0.5, 0.5),
    # 'global_ccoeff': (-0.5, 0.5),
    # 'node_percolation_random': (-0.2, 0.2),
    # 'node_percolation_targeted': (-0.2, 0.2),
    # 'frac_giant_ccomp': (-0.5, 0.5),
}

# ==============================================================================
# Statistics to consider

SCALAR_STATS = [
    # Minimum cut size (cluster)
    ('mincuts', 'sequence', 'rmse'),
    # Number of internal edges (cluster)
    ('c_edges', 'sequence', 'rmse'),
    # Degree (vertex)
    ('degree', 'sequence', 'rmse'),
    # Mixing parameter mu (vertex)
    ('mixing_mus', 'sequence', 'rmse'),
]

# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--names',
        nargs='+',
        help='Names of simulators',
        required='True',
    )
    parser.add_argument(
        '--roots',
        nargs='+',
        help='Root directories',
        required='True',
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory',
        required='True',
    )
    parser.add_argument(
        '--resolutions',
        nargs='+',
        help='Resolution to consider',
        required='True',
    )
    parser.add_argument(
        '--showfliers',
        action='store_true',
        help='Show fliers',
    )
    parser.add_argument(
        '--network-whitelist-fp',
        help='Network whitelist file path',
        required=False,
        default=None,
    )
    parser.add_argument(
        '--num-replicates',
        default=1,
        type=int,
        help='Number of replicates to consider',
    )
    parser.add_argument(
        '--ncols',
        default=3,
        type=int,
        help='Number of columns in legend for boxplot',
    )
    parser.add_argument(
        '--stats',
        nargs='+',
        help='Scalar statistics to consider',
        default=[
            stat for stat, _, _ in SCALAR_STATS
        ]
    )
    return parser.parse_args()


args = parse_args()

ncols = args.ncols
showfliers = args.showfliers
network_whitelist_fp = args.network_whitelist_fp
goal_n_replicates = args.num_replicates

# ==============================================================================

scalar_stats = [
    (stat, stat_type, distance_type)
    for stat, stat_type, distance_type in SCALAR_STATS
    if stat in args.stats
]

# ==============================================================================

network_whitelist = None \
    if network_whitelist_fp is None \
    else list(pd.read_csv(network_whitelist_fp, header=None)[0])

# ==============================================================================

assert len(args.names) == len(args.roots) == len(args.resolutions), \
    'Number of names, roots, and resolutions must be the same'

names = args.names
roots = [
    Path(root)
    for root in args.roots
]
resolutions = args.resolutions

# ==============================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print(f'Comparing {names}')

# ==============================================================================

# Get all network_ids
network_ids_each_root = [
    set(
        [
            network_dir.name
            for network_dir in root.iterdir()
            if network_dir.is_dir()
        ]
    )
    for root in roots
]
all_network_ids_raw = reduce(
    lambda x, y: x | y,
    network_ids_each_root,
)
all_network_ids = sorted(
    [
        network_id
        for network_id in all_network_ids_raw
        if network_whitelist is None or network_id in network_whitelist
    ]
)

# Get all comparison files
comp_results = dict()
successes = dict()


for network_id in all_network_ids:
    comp_results[network_id] = dict()
    successes[network_id] = dict()

    for (name, root, resolution) in zip(names, roots, resolutions):
        if not (root / network_id / resolution).exists():
            print(f'[MISSING] {root} {network_id} {resolution}')
            comp_results[network_id][name] = None
            successes[network_id][name] = 0.0
            continue

        n_successes = 0
        n_replicates = 0
        df = None
        for replicate_dir in (root / network_id / resolution).iterdir():
            replicate_id = replicate_dir.name
            if int(replicate_id) >= goal_n_replicates:
                continue

            n_replicates += 1

            found_comp_fp = False
            for comp_fn in COMP_FNS:
                comp_fp = replicate_dir / comp_fn
                if comp_fp.exists():
                    found_comp_fp = True
                    break
            if not found_comp_fp:
                continue

            n_successes += 1

            df_tmp = pd.read_csv(comp_fp)
            df_tmp['source'] = replicate_id
            df_tmp['distance'] = df_tmp['distance']

            if df is None:
                df = df_tmp
            else:
                df = pd.concat([df, df_tmp])

        r_successes = n_successes / n_replicates if n_successes > 0 else 0.0

        if n_replicates < goal_n_replicates:
            print(f'[NREPS] {root} {network_id} {resolution} {n_successes}')

        if r_successes < 1.0:
            print(f'[NSUCS] {root} {network_id} {resolution} {r_successes}')

        comp_results[network_id][name] = df
        successes[network_id][name] = r_successes

df_successes = pd.DataFrame(
    [
        [
            network_id,
            *[
                clustering_dict[name]
                for name in clustering_dict
            ]
        ]
        for network_id, clustering_dict in successes.items()
    ],
    columns=[
        'Network',
        *names,
    ]
)
df_successes.to_csv(
    output_dir / 'successes_clstats.csv',
    index=False,
    float_format='%.2f',
)

comparable_networks = [
    network_id
    for network_id in all_network_ids
    if np.array([
        successes[network_id][name] > 0
        for name in successes[network_id]
    ]).sum() == len(names)
]

agg = dict()
for network_id in comparable_networks:
    for stat, stat_type, distance_type in scalar_stats:
        agg.setdefault(
            (stat, stat_type, distance_type),
            dict(),
        )

        for name, df in comp_results[network_id].items():
            agg[(stat, stat_type, distance_type)].setdefault(
                (network_id, name),
                dict(),
            )

            if df is None:
                print(f'[MISSING] {stat} {stat_type} {
                    distance_type} {network_id} {name}')
                continue

            df_tmp = df.loc[
                (df['stat'] == stat)
                & (df['stat_type'] == stat_type)
                & (df['distance_type'] == distance_type)
            ]

            if len(df_tmp['distance'].values) > 0:
                val = df_tmp['distance'].values.mean()
            else:
                print(f'[EMPTY] {stat} {stat_type} {
                    distance_type} {network_id} {name}')
                continue

            agg[(stat, stat_type, distance_type)
                ][(network_id, name)][name] = val

# ==============================================================================

# Visualize scalar statistics
df_list = []
for (stat, stat_type, distance_type) in scalar_stats:
    if (stat, stat_type, distance_type) not in agg:
        continue
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n{resolution}'
        for sim_name, distance in data.items():
            df_list.append(
                [stat_id, sim_name, network_resolution, distance]
            )
if len(df_list) == 0:
    print('No scalar statistics to visualize')
    exit(0)
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance',
    ]
)
selection = [
    stat
    for (stat, _, _) in scalar_stats
]

nrows = len(names) // ncols + (len(names) % ncols > 0)
if len(selection) > 4:
    n_plots_per_row = (len(selection) + 1) // 2
    if n_plots_per_row == 1:
        figsize = (10, 8)
    else:
        figsize = (3 * n_plots_per_row, 8)
    fig, axes = plt.subplots(
        2,
        (len(selection) + 1) // 2,
        dpi=150,
        figsize=figsize,
    )
    for i, col in enumerate(selection):
        values = df[df['Stat'] == col]
        ax = sns.boxplot(
            x='Stat',
            y='Distance',
            hue='Simulator',
            data=values,
            ax=axes.flatten()[i],
            notch=True,
            bootstrap=10000,
            showfliers=showfliers,
        )

        if col in MINMAX_BOUNDED_SCALARS:
            lb, ub = MINMAX_BOUNDED_SCALARS[col]
            ax.set_ylim(lb, ub)

        ax.set_ylim(0.0)

        # ax.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)

        if i % ((len(selection) + 1) // 2) != 0:
            ax.set_ylabel('')
        ax.legend_.remove()
    if len(selection) % 2 == 1:
        fig.delaxes(axes.flatten()[-1])
    bbox_to_anchor = (0.5, 1.05 + 0.02 * nrows)
else:
    fig, axes = plt.subplots(
        1,
        len(selection),
        dpi=150,
        figsize=(3 * len(selection), 4),
    )
    for i, col in enumerate(selection):
        values = df[df['Stat'] == col]
        ax = sns.boxplot(
            x='Stat',
            y='Distance',
            hue='Simulator',
            data=values,
            ax=axes.flatten()[i],
            notch=True,
            bootstrap=10000,
            showfliers=showfliers,
        )

        if col in MINMAX_BOUNDED_SCALARS:
            lb, ub = MINMAX_BOUNDED_SCALARS[col]
            ax.set_ylim(lb, ub)

        ax.set_ylim(0.0)

        # ax.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)

        ax.legend_.remove()

    bbox_to_anchor = (0.5, 1.1 + 0.03 * nrows)
handles, labels = axes.flatten()[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=ncols,
    bbox_to_anchor=bbox_to_anchor,
    fancybox=True,
)
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_clstats.pdf', bbox_inches='tight')
