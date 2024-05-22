import os
import glob
import json
from pathlib import Path

import click
import scipy
import numpy as np
import pandas as pd

from constants import (
    NODE_ORDER_FN,
    COMM_ORDER_FN,
    OUTLIER_ORDER_FN,
    NODE_DISTR_STATS,
    COMM_DISTR_STATS,
    OUTLIER_DISTR_STATS
)


def parse_distribution(distribution_path):
    distribution_arr = []
    with open(distribution_path, "r") as f:
        for line in f:
            distribution_arr.append(float(line))
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


def sequence_distance(input_sequence, replicate_sequence):
    try:
        d = np.linalg.norm(
            input_sequence - replicate_sequence, ord=1)
        d = d / len(input_sequence)
    except Exception as e:
        print(f"error: {e}")
        d = np.nan

    try:
        d2 = np.linalg.norm(
            input_sequence - replicate_sequence, ord=2)
    except Exception as e:
        print(f"error: {e}")
        d2 = np.nan

    return d, d2


def compare_scalars(
    network_1_folder,
    network_2_folder,
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    network_1_fns = \
        glob.glob(f"{network_1_folder}/*.distribution")
    network_1_stats = dict()
    for fn in network_1_fns:
        name = Path(fn).stem
        network_1_stats[name] = parse_distribution(fn)

    network_2_fns = \
        glob.glob(f"{network_2_folder}/*.distribution")
    network_2_stats = dict()
    for fn in network_2_fns:
        name = Path(fn).stem
        network_2_stats[name] = parse_distribution(fn)

    common_stats = set(
        network_1_stats.keys()
    ).intersection(
        network_2_stats.keys()
    )

    diff_dict = dict()
    for name in common_stats:
        diff_dict[name] = dict()

        network_1_distr = network_1_stats[name]
        network_2_distr = network_2_stats[name]

        for dist_type in ['ks', 'emd']:
            try:
                diff = distribution_distance(
                    network_1_distr,
                    network_2_distr,
                    dist_type,
                )
            except Exception as e:
                print(f"[ERROR] ({dist_type}) {e}")
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
    input_network_folder,
    input_replicates_folder,
    output_folder,
) -> None:
    # TODO: finish this
    return None


@ click.command()
@ click.option("--network-1-folder", required=True, type=click.Path(exists=True), help="Input network stats folder")
@ click.option("--network-2-folder", required=True, type=click.Path(exists=True), help="Input synthetic stats folder")
@ click.option("--output-file", required=True, type=click.Path(dir_okay=False, writable=True), help="Ouput folder to save the comparison results")
@ click.option("--is-node-sequence", is_flag=True, help="Whether node distributions can be treated as sequences (default: False)")
@ click.option("--is-cluster-sequence", is_flag=True, help="Whether cluster distributions can be treated as sequences (default: False)")
def compare_stats(
    network_1_folder,
    network_2_folder,
    output_file,
    is_node_sequence,
    is_cluster_sequence,
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

    # compare_sequences(
    #     network_1_folder,
    #     network_2_folder,
    #     output_folder,
    # )

    df = pd.DataFrame(
        data=df_list,
        columns=[
            'stat',
            'stat_type',
            'distance_type',
            'distance',
        ]
    )
    print(df)
    # df.to_csv(output_file, index=False, float_format='%.4f')


if __name__ == "__main__":
    compare_stats()
