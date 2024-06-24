import argparse
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

COMP_FN = 'compare_output.csv'

stats = [
    ('mincuts', 'distribution', 'ks'),
    ('diameter', 'scalar', 'rpd'),
    ('mixing_mus', 'distribution', 'ks'),
    ('global_ccoeff', 'scalar', 'rpd'),
    ('local_ccoeff', 'scalar', 'rpd'),
    ('degree', 'distribution', 'ks'),
    ('mixing_xi', 'scalar', 'rpd'),
    ('c_edges', 'distribution', 'ks'),
    ('concomp_sizes', 'distribution', 'ks'),
    ('n_concomp', 'scalar', 'rpd'),
]


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
    return parser.parse_args()


args = parse_args()
assert len(args.names) == len(args.roots)

names = args.names
roots = [
    Path(root)
    for root in args.roots
]
output_dir = Path(args.output_dir)

# Expected structure:
# root
# ├── {network_id}
# │   ├── leiden{resolution}
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


output_dir.mkdir(exist_ok=True, parents=True)

network_ids = [
    set(x.name for x in root.iterdir())
    for root in roots
]

all_network_ids = reduce(
    lambda x, y: x | y,
    network_ids,
)

all_resolutions = dict()
for network_id in all_network_ids:
    resolutions = [
        set(
            x.name.replace('leiden', '')
            for x in (root / network_id).iterdir()
        )
        if (root / network_id).exists()
        else set()
        for root in roots
    ]

    all_resolutions[network_id] = reduce(
        lambda x, y: x | y,
        resolutions,
    )

all_replicates = dict()
for network_id in all_network_ids:
    all_replicates[network_id] = dict()
    for resolution in all_resolutions[network_id]:
        replicate_ids = [
            set(
                x.name
                for x in (root / network_id / f'leiden{resolution}').iterdir()
            )
            if (root / network_id / f'leiden{resolution}').exists()
            else set()
            for root in roots
        ]

        all_replicates[network_id][resolution] = reduce(
            lambda x, y: x | y,
            replicate_ids,
        )


comp_results = dict()
successes = dict()

for network_id in all_network_ids:
    successes[network_id] = dict()
    comp_results[network_id] = dict()

    for resolution in all_resolutions[network_id]:
        successes[network_id][resolution] = []
        comp_results[network_id][resolution] = []

        for root in roots:
            n_successes = 0
            comp = []

            n_replicates = len(all_replicates[network_id][resolution])

            df = None
            for replicate_id in all_replicates[network_id][resolution]:
                comp_fp = root / network_id / \
                    f'leiden{resolution}' / replicate_id / COMP_FN

                if not comp_fp.exists():
                    continue

                n_successes += 1

                df_tmp = pd.read_csv(comp_fp)
                df_tmp['source'] = replicate_id
                df_tmp['distance'] = df_tmp['distance']

                if df is None:
                    df = df_tmp
                else:
                    df = pd.concat([df, df_tmp])

            ratio_successes = n_successes / n_replicates
            successes[network_id][resolution].append(ratio_successes)
            comp_results[network_id][resolution].append(df)

#

df_successes = pd.DataFrame(
    [
        [
            network_id,
            resolution,
            *successes[network_id][resolution],
        ]
        for network_id in all_network_ids
        for resolution in all_resolutions[network_id]
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

#

comparable_pairs = [
    (network_id, resolution)
    for network_id in all_network_ids
    for resolution in all_resolutions[network_id]
    if (np.array(successes[network_id][resolution]) > 0).sum() > 1
]

# output_tables_dir = output_dir / 'tables'
# if not output_tables_dir.exists():
#     output_tables_dir.mkdir(parents=True)
# else:
#     for fp in output_tables_dir.iterdir():
#         fp.unlink()

agg = dict()
for network_id, resolution in comparable_pairs:
    dfs = comp_results[network_id][resolution]
    for name, df in zip(names, dfs):
        if df is None:
            continue

        for stat, stat_type, distance_type in stats:
            df_tmp = df.loc[
                (df['stat'] == stat)
                & (df['stat_type'] == stat_type)
                & (df['distance_type'] == distance_type)
            ]

            agg.setdefault(
                (stat, stat_type, distance_type),
                dict(),
            ).setdefault(
                (network_id, resolution),
                dict(),
            )[name] = \
                df_tmp['distance'].values.mean()

distr_stats = [
    (stat, stat_type, distance_type)
    for (stat, stat_type, distance_type) in stats
    if stat_type == 'distribution'
]

df_list = []
for (stat, stat_type, distance_type) in distr_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n$r=0{resolution}$'
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
        'Distance',
    ]
)
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(2 * len(distr_stats), 5))
ax = sns.boxplot(
    x='Stat',
    y='Distance',
    hue='Simulator',
    data=df,
)
ax.set_ylim(-0.1, 1.1)
plt.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_distr.png')

scalar_stats = [
    (stat, stat_type, distance_type)
    for (stat, stat_type, distance_type) in stats
    if stat_type == 'scalar'
]
df_list_nonneg = []
df_list_nonpos = []
for (stat, stat_type, distance_type) in scalar_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n$r=0{resolution}$'
        for sim_name, distance in data.items():
            if distance >= 0:
                df_list_nonneg.append(
                    [stat_id, sim_name, network_resolution, distance]
                )
            elif distance < 0:
                df_list_nonpos.append(
                    [stat_id, sim_name, network_resolution, distance]
                )

fig, ax = plt.subplots(2, 1, dpi=150, figsize=(2 * len(scalar_stats), 5))
df_nonneg = pd.DataFrame(
    df_list_nonneg,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance',
    ]
)
sns.boxplot(
    x='Stat',
    y='Distance',
    hue='Simulator',
    data=df_nonneg,
    ax=ax[0],
)
ax[0].set_ylim(-0.0, 1.1)
# ax[0].axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
# ax[0].set_title('Non-negative distances')
ax[0].set_xlabel('')
ax[0].set_xticklabels([])
df_nonpos = pd.DataFrame(
    df_list_nonpos,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance',
    ]
)
sns.boxplot(
    x='Stat',
    y='Distance',
    hue='Simulator',
    data=df_nonpos,
    ax=ax[1],
)
ax[1].set_ylim(-1.1, 0.0)
# ax[1].axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
# ax[1].set_title('Non-positive distances')
ax[1].set_xlabel('')
ax[1].legend_.remove()
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_scalar.png')

df_list = []
for (stat, stat_type, distance_type) in scalar_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n$r=0{resolution}$'
        for sim_name, distance in data.items():
            if distance not in [-1, 1]:
                df_list.append(
                    [stat_id, sim_name, network_resolution, distance]
                )
df = pd.DataFrame(
    df_list,
    columns=[
        'Stat',
        'Simulator',
        'Network',
        'Distance',
    ]
)
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(2 * len(distr_stats), 5))
ax = sns.boxplot(
    x='Stat',
    y='Distance',
    hue='Simulator',
    data=df,
)
ax.set_ylim(-1.1, 1.1)
plt.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_scalar_joined_remove.png')

df_list = []
for (stat, stat_type, distance_type) in scalar_stats:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n$r=0{resolution}$'
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
        'Distance',
    ]
)
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(2 * len(scalar_stats), 5))
ax = sns.boxplot(
    x='Stat',
    y='Distance',
    hue='Simulator',
    data=df,
)
ax.set_ylim(-1.1, 1.1)
plt.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
fig.tight_layout()
fig.savefig(output_dir / 'boxplot_scalar_joined.png')

scalar_stats_potential_0 = [
    (stat, stat_type, distance_type)
    for (stat, stat_type, distance_type) in stats
    if stat in ['global_ccoeff', 'local_ccoeff']
]
df_list = []
for (stat, stat_type, distance_type) in scalar_stats_potential_0:
    sim_dict = agg[(stat, stat_type, distance_type)]
    stat_id = f'{stat}'
    for (network_id, resolution), data in sim_dict.items():
        network_resolution = f'{network_id}\n$r=0{resolution}$'
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
        'Distance',
    ]
)
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(
    2 * len(scalar_stats_potential_0), 5))
ax = sns.violinplot(
    x='Stat',
    y='Distance',
    hue='Simulator',
    data=df,
)
ax.set_ylim(-1.1, 1.1)
plt.axhline(y=0, color='r', linestyle='dashed', linewidth=0.5)
fig.tight_layout()
fig.savefig(output_dir / 'violinplot_scalar_joined.png')
