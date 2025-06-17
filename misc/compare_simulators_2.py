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
    'deg_assort': (-0.4, 0.4),
    'local_ccoeff': (-0.5, 0.5),
    'global_ccoeff': (-0.5, 0.5),
    'mixing_xi': (-0.15, 0.15),
}


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
        '--resolution',
        help='Resolution to consider',
        required='True',
    )
    parser.add_argument(
        '--show-fliers',
        action='store_true',
        help='Show fliers',
    )
    parser.add_argument(
        '--with-outliers',
        action='store_true',
        help='Include outliers',
    )
    parser.add_argument(
        '--network-whitelist-fp',
        help='Network whitelist file path',
        required=False,
        default=None,
    )
    parser.add_argument(
        '--num-replicates',
        help='Number of replicates to consider',
    )
    return parser.parse_args()


args = parse_args()

RESOLUTIONS = [args.resolution]
SHOWFLIERS = args.show_fliers

NETWORK_WHITELIST = None \
    if args.network_whitelist_fp is None \
    else list(pd.read_csv(args.network_whitelist_fp, header=None)[0])

GOAL_N_REPLICATES = int(args.num_replicates)

cluster_seq_stats = [
    # Minimum cut size (cluster)
    ('mincuts', 'sequence', 'rmse'),
    # Number of internal edges (cluster)
    ('c_edges', 'sequence', 'rmse'),
]
vertex_seq_stats = [
    # Degree (vertex)
    ('degree', 'sequence', 'rmse'),
    # Mixing parameter mu (vertex)
    ('mixing_mus', 'sequence', 'rmse'),
] if not args.with_outliers else [
    # Degree (vertex)
    ('degree', 'sequence', 'rmse'),
]
distr_stats = [
    # Minimum cut size (cluster)
    ('mincuts', 'distribution', 'ks'),
    # Number of internal edges (cluster)
    ('c_edges', 'distribution', 'ks'),

    # Degree (vertex)
    ('degree', 'distribution', 'ks'),
    # Mixing parameter mu (vertex)
    ('mixing_mus', 'distribution', 'ks'),
] if not args.with_outliers else [
    # Degree (vertex)
    ('degree', 'distribution', 'ks'),
]
bounded_scalar_stats = [
    # Degree assortativity
    ('deg_assort', 'scalar', 'abs_diff'),
    # Mean local clustering coefficient
    ('local_ccoeff', 'scalar', 'abs_diff'),
    # Global clustering coefficient
    ('global_ccoeff', 'scalar', 'abs_diff'),
    # Mixing parameter xi
    ('mixing_xi', 'scalar', 'abs_diff'),
    # Node percolation profile (random removal)
    # TODO: Not implemented
    # Node percolation profile (targeted removal)
    # TODO: Not implemented
    # Fraction of nodes in the largest component
    # TODO: Not implemented
] if not args.with_outliers else [
    # Degree assortativity
    ('deg_assort', 'scalar', 'abs_diff'),
    # Mean local clustering coefficient
    ('local_ccoeff', 'scalar', 'abs_diff'),
    # Global clustering coefficient
    ('global_ccoeff', 'scalar', 'abs_diff'),
    # Node percolation profile (random removal)
    # TODO: Not implemented
    # Node percolation profile (targeted removal)
    # TODO: Not implemented
    # Fraction of nodes in the largest component
    # TODO: Not implemented
]
positive_scalar_stats = [
    # Number of edges
    ('n_edges', 'scalar', 'rel_diff'),
    # Number of connected components
    ('n_concomp', 'scalar', 'rel_diff'),
    # Pseudo-diameter
    ('diameter', 'scalar', 'rel_diff'),
    # Mean k-core
    # TODO: Not implemented
    # Leading eigenvalue of adjacency matrix
    # TODO: Not implemented
    # Leading eigenvalue of Hashimoto matrix
    # TODO: Not implemented
    # Characteristic time of a random walk
    # TODO: Not implemented
]

stats = sum(
    [
        cluster_seq_stats,
        vertex_seq_stats,
        distr_stats,
        bounded_scalar_stats,
        positive_scalar_stats,
    ],
    [],
)

assert len(args.names) == len(args.roots)

names = args.names
roots = [
    Path(root)
    for root in args.roots
]
output_dir = Path(args.output_dir)

output_dir.mkdir(exist_ok=True, parents=True)

print(f'Comparing {names} at {RESOLUTIONS}')

# Collect all network IDs
network_ids = [
    set(x.name for x in root.iterdir())
    for root in roots
]

all_network_ids = reduce(
    lambda x, y: x | y,
    network_ids,
)

# Collect all replicates
all_replicates = dict()
for network_id in all_network_ids:
    all_replicates[network_id] = dict()
    for resolution in RESOLUTIONS:
        replicate_ids = [
            set(
                x.name
                for x in (root / network_id / f'{resolution}').iterdir()
            )
            if (root / network_id / f'{resolution}').exists()
            else set()
            for root in roots
        ]

        all_replicates[network_id][resolution] = reduce(
            lambda x, y: x | y,
            replicate_ids,
        )

        all_replicates[network_id][resolution] = [
            x
            for x in all_replicates[network_id][resolution]
            if int(x) < GOAL_N_REPLICATES
        ]


all_network_ids = [
    network_id
    for network_id in all_network_ids
    if any(
        len(all_replicates[network_id][resolution]) > 0
        for resolution in RESOLUTIONS
    )
    and network_id in (
        NETWORK_WHITELIST
        if NETWORK_WHITELIST is not None
        else all_network_ids
    )
]

comp_results = dict()
successes = dict()

for network_id in all_network_ids:
    successes[network_id] = dict()
    comp_results[network_id] = dict()

    for resolution in RESOLUTIONS:
        successes[network_id][resolution] = []
        comp_results[network_id][resolution] = []

        for root in roots:
            n_successes = 0
            comp = []

            n_replicates = 0

            df = None

            for replicate_id in all_replicates[network_id][resolution]:
                comp_root_fp = root / network_id / \
                    f'{resolution}' / replicate_id

                if not comp_root_fp.exists():
                    continue

                n_replicates += 1

                found_comp_fp = False
                for comp_fn in COMP_FNS:
                    comp_fp = comp_root_fp / comp_fn

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

            ratio_successes = n_successes / n_replicates if n_successes > 0 else 0.0

            if n_replicates < GOAL_N_REPLICATES:
                print(f'[NREPS] {root} {network_id} {
                      resolution} {n_successes}')

            if ratio_successes < 1.0:
                print(f'[NSUCS] {root} {network_id} {
                      resolution} {ratio_successes}')

            successes[network_id][resolution].append(ratio_successes)
            comp_results[network_id][resolution].append(df)

df_successes = pd.DataFrame(
    [
        [
            network_id,
            resolution,
            *successes[network_id][resolution],
        ]
        for network_id in all_network_ids
        for resolution in RESOLUTIONS
    ],
    columns=[
        'Network',
        'Resolution',
        *names,
    ],
)
df_successes = df_successes.sort_values(
    by=['Network', 'Resolution'],
)
df_successes.to_csv(
    output_dir / 'successes.csv',
    index=False,
    float_format='%.2f',
)

comparable_pairs = [
    (network_id, resolution)
    for network_id in all_network_ids
    for resolution in RESOLUTIONS
    if (np.array(successes[network_id][resolution]) > 0).sum() == len(names)
]

agg = dict()
for network_id, resolution in comparable_pairs:
    dfs = comp_results[network_id][resolution]
    for stat, stat_type, distance_type in stats:
        agg.setdefault(
            (stat, stat_type, distance_type),
            dict(),
        ).setdefault(
            (network_id, resolution),
            dict(),
        )

        for name, df in zip(names, dfs):
            if df is None:
                print(f'[MISSING] {stat} {stat_type} {
                      distance_type} {network_id} {resolution} {name}')
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
                      distance_type} {network_id} {resolution} {name}')
                continue

            agg[(stat, stat_type, distance_type)][(
                network_id, resolution)][name] = val

# Visualize distribution statistics
df_list = []
for (stat, stat_type, distance_type) in distr_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n{resolution}'
        for sim_name, distance in data.items():
            df_list.append(
                [stat_id, sim_name, network_resolution, distance]
            )
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance (KS)',
    ]
)
fig, ax = plt.subplots(
    1, 1,
    dpi=150,
    figsize=(3 * len(distr_stats), 5)
)
ax = sns.boxplot(
    x='Stat',
    y='Distance (KS)',
    hue='Simulator',
    data=df,
    showfliers=SHOWFLIERS,
)
ax.set_ylim(0., 1.)
# plt.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_distr.pdf')

# Visualize positive scalar statistics
df_list = []
for (stat, stat_type, distance_type) in positive_scalar_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n{resolution}'
        for sim_name, distance in data.items():
            df_list.append([
                stat_id,
                sim_name,
                network_resolution,
                distance,
            ])
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance (SRD)',
    ]
)
selection = [
    stat
    for (stat, _, _) in positive_scalar_stats
]
fig, axes = plt.subplots(
    1, len(selection),
    dpi=150,
    figsize=(3 * len(selection), 5),
)
for i, col in enumerate(selection):
    values = df[df['Stat'] == col]
    ax = sns.boxplot(
        x='Stat',
        y='Distance (SRD)',
        hue='Simulator',
        data=values,
        ax=axes.flatten()[i],
        showfliers=SHOWFLIERS,
    )
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        ncol=1,
        fancybox=True,
    )

    if SHOWFLIERS:
        lb = values['Distance (SRD)'].quantile(1-Q)
        ub = values['Distance (SRD)'].quantile(Q)
        ax.set_ylim(lb, ub)

    ax.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)

    if i != 0:
        ax.set_ylabel('')
    ax.legend_.remove()
    # if i != len(selection) - 1:
    #     ax.legend_.remove()
# ax.legend_.remove()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(names))
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_positive_scalar.pdf')

# Visualize bounded scalar statistics
df_list = []
for (stat, stat_type, distance_type) in bounded_scalar_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n{resolution}'
        for sim_name, distance in data.items():
            df_list.append(
                [stat_id, sim_name, network_resolution, distance]
            )
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance (SAD)',
    ]
)
selection = [
    stat
    for (stat, _, _) in bounded_scalar_stats
]
fig, axes = plt.subplots(1, len(selection), dpi=150,
                         figsize=(3 * len(selection), 5))
for i, col in enumerate(selection):
    values = df[df['Stat'] == col]
    ax = sns.boxplot(
        x='Stat',
        y='Distance (SAD)',
        hue='Simulator',
        data=values,
        ax=axes.flatten()[i],
        showfliers=SHOWFLIERS,
    )
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              ncol=1, fancybox=True)

    lb, ub = MINMAX_BOUNDED_SCALARS[col]
    ax.set_ylim(lb - 0.1, ub + 0.1)
    ax.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)

    if i != 0:
        ax.set_ylabel('')
    if i != len(selection) - 1:
        ax.legend_.remove()
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_bounded_scalar.pdf')

# Visualize cluster sequence statistics
df_list = []
for (stat, stat_type, distance_type) in cluster_seq_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n{resolution}'
        for sim_name, distance in data.items():
            df_list.append(
                [stat_id, sim_name, network_resolution, distance]
            )
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance (RMSE)',
    ]
)
selection = [
    stat
    for (stat, _, _) in cluster_seq_stats
]
fig, axes = plt.subplots(1, len(selection), dpi=150,
                         figsize=(3 * len(selection), 5))
for i, col in enumerate(selection):
    values = df[df['Stat'] == col]
    ax = sns.boxplot(
        x='Stat',
        y='Distance (RMSE)',
        hue='Simulator',
        data=values,
        ax=axes.flatten()[i],
        showfliers=SHOWFLIERS,
    )
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              ncol=1, fancybox=True)

    if SHOWFLIERS:
        ub = values['Distance (RMSE)'].quantile(Q)
        ax.set_ylim(0.0, ub)

    if i != 0:
        ax.set_ylabel('')
    if i != len(selection) - 1:
        ax.legend_.remove()
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_cluster_seq.pdf')

# Visualize vertex sequence statistics
df_list = []
for (stat, stat_type, distance_type) in vertex_seq_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n{resolution}'
        for sim_name, distance in data.items():
            df_list.append(
                [stat_id, sim_name, network_resolution, distance]
            )
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance (RMSE)',
    ]
)
selection = [
    stat
    for (stat, _, _) in vertex_seq_stats
]
fig, axes = plt.subplots(1, len(selection), dpi=150,
                         figsize=(3 * len(selection), 5))
for i, col in enumerate(selection):
    values = df[df['Stat'] == col]
    ax = sns.boxplot(
        x='Stat',
        y='Distance (RMSE)',
        hue='Simulator',
        data=values,
        ax=axes.flatten()[i] if len(selection) > 1 else axes,
        showfliers=SHOWFLIERS,
    )
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              ncol=1, fancybox=True)

    if SHOWFLIERS:
        ub = values['Distance (RMSE)'].quantile(Q)
        ax.set_ylim(0.0, ub)

    if i != 0:
        ax.set_ylabel('')
    if i != len(selection) - 1:
        ax.legend_.remove()
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_vertex_seq.pdf')
