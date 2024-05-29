import os
import glob
import json
from pathlib import Path
from typing import List, Any

import click
import scipy
import numpy as np
import pandas as pd

NODE_ORDER_FN = 'node_ordering.idx'
COMM_ORDER_FN = 'cluster_ordering.idx'
OUTLIER_ORDER_FN = 'outlier_ordering.idx'

DEGR_DISTR = 'degree'
O_DEGR_DISTR = 'o_deg'
CSIZE_DISTR = 'c_size'
MCS_DIST = 'mincuts'
O_PART_COEFF_DISTR = 'o_participation_coeffs'
PART_COEFF_DISTR = 'participation_coeffs'

NODE_DISTR_STATS = [DEGR_DISTR, PART_COEFF_DISTR]
COMM_DISTR_STATS = [CSIZE_DISTR, MCS_DIST]
OUTLIER_DISTR_STATS = [O_DEGR_DISTR, O_PART_COEFF_DISTR]


def parse_distribution(distribution_path):
    distribution_arr = []
    with open(distribution_path, "r") as f:
        df = pd.read_csv(f, header=None)
        distribution_arr = df[0].values
    return distribution_arr


def parse_json(stats_path):
    with open(stats_path, "r") as f:
        return json.load(f)


def scalar_distance(input_scalar, replicate_scalar, dist_type):
    if dist_type == 'abs_diff':
        d = abs(input_scalar - replicate_scalar)
    elif dist_type == 'rel_diff':
        d = abs(input_scalar - replicate_scalar) / input_scalar
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")
    return d


def distribution_distance(input_distribution, replicate_distribution, dist_type):
    if dist_type == 'ks':
        d = scipy.stats.ks_2samp(
            input_distribution,
            replicate_distribution,
        ).statistic
    elif dist_type == 'emd':
        d = scipy.stats.wasserstein_distance(
            input_distribution,
            replicate_distribution,
        )
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")
    return d


def sequence_distance(seq_1, seq_2, dist_type):
    if dist_type == 'l1':
        d = np.linalg.norm(seq_1 - seq_2, ord=1)
    elif dist_type == 'mean_l1':
        d = np.linalg.norm(seq_1 - seq_2, ord=1)
        d = d / len(seq_1)
    elif dist_type == 'l2':
        d = np.linalg.norm(seq_1 - seq_2, ord=2)
    elif dist_type == 'rmse':
        d = np.sqrt(np.mean((seq_1 - seq_2) ** 2))
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")

    return d


def compare_scalars(
    network_1_folder,
    network_2_folder,
) -> List[List[Any]]:
    network_1_stats = parse_json(f"{network_1_folder}/stats.json")
    network_2_stats = parse_json(f"{network_2_folder}/stats.json")

    common_stats = set(
        network_1_stats.keys()
    ).intersection(
        network_2_stats.keys()
    )

    diff_dict = dict()
    for name in common_stats:
        diff_dict[name] = dict()

        for dist_type in ['abs_diff', 'rel_diff']:
            try:
                diff = scalar_distance(
                    network_1_stats[name],
                    network_2_stats[name],
                    dist_type,
                )
            except Exception as e:
                print(f"[ERROR] ({dist_type}) {e}")
                diff = np.nan

            diff_dict[name][dist_type] = diff

    df_lists = [
        [
            name,
            'scalar',
            diff_type,
            diff,
        ]
        for name, diff_dict in diff_dict.items()
        for diff_type, diff in diff_dict.items()
    ]
    return df_lists


def compare_distributions(
    network_1_folder,
    network_2_folder,
) -> List[List[Any]]:
    network_1_fns = \
        glob.glob(f"{network_1_folder}/*.distribution")
    network_1_stats = dict()
    for fn in network_1_fns:
        try:
            name = Path(fn).stem
            network_1_stats[name] = parse_distribution(fn)
        except Exception as e:
            print(f"[ERROR] ({name}) {e}")
            network_1_stats[name] = []

    network_2_fns = \
        glob.glob(f"{network_2_folder}/*.distribution")
    network_2_stats = dict()
    for fn in network_2_fns:
        try:
            name = Path(fn).stem
            network_2_stats[name] = parse_distribution(fn)
        except Exception as e:
            print(f"[ERROR] ({name}) {e}")
            network_2_stats[name] = []

    common_stats = set(
        network_1_stats.keys()
    ).intersection(
        network_2_stats.keys()
    )

    diff_dict = dict()
    for name in common_stats:
        network_1_distr = network_1_stats[name]
        network_2_distr = network_2_stats[name]

        diff_dict[name] = dict()
        for dist_type in ['ks', 'emd']:
            try:
                diff = distribution_distance(
                    network_1_distr,
                    network_2_distr,
                    dist_type,
                )
            except Exception as e:
                print(f"[ERROR] ({name}) ({dist_type}) {e}")
                diff = np.nan

            diff_dict[name][dist_type] = diff

    df_lists = [
        [
            name,
            'distribution',
            diff_type,
            diff,
        ]
        for name, diff_dict in diff_dict.items()
        for diff_type, diff in diff_dict.items()
    ]
    return df_lists


def compare_sequences(
    network_1_folder,
    network_2_folder,
) -> List[List[Any]]:
    network_1_fns = \
        glob.glob(f"{network_1_folder}/*.distribution")
    network_1_stats = dict()
    for fn in network_1_fns:
        try:
            name = Path(fn).stem
            network_1_stats[name] = parse_distribution(fn)
        except Exception as e:
            print(f"[ERROR] ({name}) {e}")
            network_1_stats[name] = []

    network_1_ids = {
        'node':
            pd.read_csv(f"{network_1_folder}/{NODE_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{network_1_folder}/{NODE_ORDER_FN}")
            else None,
        'comm':
            pd.read_csv(f"{network_1_folder}/{COMM_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{network_1_folder}/{COMM_ORDER_FN}")
            else None,
        'o_node':
            pd.read_csv(f"{network_1_folder}/{OUTLIER_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{network_1_folder}/{OUTLIER_ORDER_FN}")
            else None,
    }

    network_2_fns = \
        glob.glob(f"{network_2_folder}/*.distribution")
    network_2_stats = dict()
    for fn in network_2_fns:
        try:
            name = Path(fn).stem
            network_2_stats[name] = parse_distribution(fn)
        except Exception as e:
            print(f"[ERROR] ({name}) {e}")
            network_2_stats[name] = []

    network_2_ids = {
        'node':
            pd.read_csv(f"{network_2_folder}/{NODE_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{network_2_folder}/{NODE_ORDER_FN}")
            else None,
        'comm':
            pd.read_csv(f"{network_2_folder}/{COMM_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{network_2_folder}/{COMM_ORDER_FN}")
            else None,
        'o_node':
            pd.read_csv(f"{network_2_folder}/{OUTLIER_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{network_2_folder}/{OUTLIER_ORDER_FN}")
            else None,
    }

    common_stats = set(
        network_1_stats.keys()
    ).intersection(
        network_2_stats.keys()
    )

    diff_dict = dict()
    for name in common_stats:
        if name in NODE_DISTR_STATS:
            df_1_ids = network_1_ids['node']
            df_2_ids = network_2_ids['node']
        elif name in COMM_DISTR_STATS:
            df_1_ids = network_1_ids['comm']
            df_2_ids = network_2_ids['comm']
        elif name in OUTLIER_DISTR_STATS:
            df_1_ids = network_1_ids['o_node']
            df_2_ids = network_2_ids['o_node']
        else:
            continue
        assert df_1_ids is not None and df_2_ids is not None

        df_1_values = pd.DataFrame(network_1_stats[name], columns=['1'])
        assert len(df_1_ids) == len(df_1_values)
        df_1 = pd.concat([df_1_ids, df_1_values], axis=1)

        df_2_values = pd.DataFrame(network_2_stats[name], columns=['2'])
        assert len(df_2_ids) == len(df_2_values)
        df_2 = pd.concat([df_2_ids, df_2_values], axis=1)

        # TODO: Handle cases where there are outliers. For now, just ignore them.
        # assert len(df_1) == len(df_2), f"{len(df_1)} != {len(df_2)}"
        df_joined = pd.merge(df_1, df_2, on='id', how='inner')
        # assert len(df_joined) == len(df_1) and len(df_joined) == len(df_2)

        network_1_seq = df_joined['1'].values
        network_2_seq = df_joined['2'].values

        diff_dict[name] = dict()
        for dist_type in ['l1', 'mean_l1', 'l2', 'rmse']:
            try:
                diff = sequence_distance(
                    network_1_seq,
                    network_2_seq,
                    dist_type,
                )
            except Exception as e:
                print(f"[ERROR] ({dist_type}) {e}")
                diff = np.nan

            diff_dict[name][dist_type] = diff

    df_lists = [
        [
            name,
            'sequence',
            diff_type,
            diff,
        ]
        for name, diff_dict in diff_dict.items()
        for diff_type, diff in diff_dict.items()
    ]
    return df_lists


@ click.command()
@ click.option(
    "--network-1-folder",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
    ),
    help="Input 1st network stats folder",
)
@ click.option(
    "--network-2-folder",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
    ),
    help="Input 2nd network stats folder",
)
@ click.option(
    "--output-file",
    required=True,
    type=click.Path(
        dir_okay=False,
        writable=True,
    ),
    help="Ouput folder to save the comparison results",
)
@ click.option(
    "--is-compare-sequence",
    is_flag=True,
    help="Whether to compare between distributions as though they are sequences",
)
def compare_stats(
    network_1_folder,
    network_2_folder,
    output_file,
    is_compare_sequence,
) -> None:
    df_list = []

    scalars_df_list = \
        compare_scalars(
            network_1_folder,
            network_2_folder,
        )
    df_list.extend(scalars_df_list)

    distributions_df_list = \
        compare_distributions(
            network_1_folder,
            network_2_folder,
        )
    df_list.extend(distributions_df_list)

    # TODO: handle when only node (and not cluster) has 1-1 correspondence (and vice versa)
    if is_compare_sequence:
        sequence_df_list = \
            compare_sequences(
                network_1_folder,
                network_2_folder,
            )
        df_list.extend(sequence_df_list)

    df = pd.DataFrame(
        data=df_list,
        columns=[
            'stat',
            'stat_type',
            'distance_type',
            'distance',
        ]
    )
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    compare_stats()
