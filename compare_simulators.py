import argparse
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

COMP_FN = 'compare_output.csv'


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

comparable_pairs = {
    (network_id, resolution)
    for network_id in all_network_ids
    for resolution in all_resolutions[network_id]
    if (np.array(successes[network_id][resolution]) > 0).sum() > 1
}

output_tables_dir = output_dir / 'tables'
if not output_tables_dir.exists():
    output_tables_dir.mkdir(parents=True)
else:
    for fp in output_tables_dir.iterdir():
        fp.unlink()

for network_id, resolution in comparable_pairs:
    dfs = comp_results[network_id][resolution]
    agg_df = None
    for name, df in zip(names, dfs):
        if df is None:
            continue

        grouped_df = df.groupby([
            'stat',
            'stat_type',
            'distance_type',
        ]).agg({
            'distance': ['count', 'mean', 'std'],
        })

        grouped_df.columns = grouped_df.columns.map('_'.join)
        grouped_df = grouped_df.reset_index()
        grouped_df = grouped_df.rename(columns={
            'distance_count': f'distance_count_{name}',
            'distance_mean': f'distance_mean_{name}',
            'distance_std': f'distance_std_{name}',
        })

        if agg_df is None:
            agg_df = grouped_df
        else:
            agg_df = pd.merge(
                agg_df,
                grouped_df,
                on=['stat', 'stat_type', 'distance_type'],
            )

    print(f'{network_id} {resolution}')

    agg_df.to_csv(
        output_tables_dir / f'{network_id}_{resolution}.csv',
        index=False,
    )
