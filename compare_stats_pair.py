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


def distribution_distance(input_distribution, replicate_distribution):
    try:
        d = scipy.stats.ks_2samp(
            input_distribution,
            replicate_distribution,
        ).statistic
    except Exception as e:
        print(f"error: {e}")
        d = np.nan
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

        diff = abs(network_1_stats[name] - network_2_stats[name])
        diff_dict[name]['abs_diff'] = diff

        # relative_diff = diff / network_1_stats[name]
        # diff_dict[name]['rel_diff'] = relative_diff

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
) -> None:
    # TODO: not done yet
    current_distribution_arr = \
        glob.glob(f"{network_1_folder}/*.distribution")
    distribution_name_arr = []
    for current_distribution_file in current_distribution_arr:
        distribution_name = Path(current_distribution_file).stem
        distribution_name_arr.append(distribution_name)
        input_distribution_stat_dict[distribution_name] = parse_distribution(
            current_distribution_file)

    for current_replicate_index in range(num_replicates):
        replicates_distribution_stat_dict[current_replicate_index] = {}
        for current_distribution_name in distribution_name_arr:
            current_distribution_file = f"{
                replicate_folder_arr[current_replicate_index]}/{current_distribution_name}.distribution"
            replicates_distribution_stat_dict[current_replicate_index][current_distribution_name] = parse_distribution(
                current_distribution_file)

    diff_dict = dict()
    for distribution_name in distribution_name_arr:
        print(f"evaluating {distribution_name} k-s stat")
        diff_dict[distribution_name] = dict()
        for replicate_id in range(num_replicates):
            input_distribution = input_distribution_stat_dict[distribution_name]
            replicate_distribution = \
                replicates_distribution_stat_dict[replicate_id][distribution_name]
            diff = distribution_distance(
                input_distribution,
                replicate_distribution,
            )
            diff_dict[distribution_name][replicate_id] = diff
            print(f"replicate {replicate_id}: {diff}")

    # Write results to csv
    df_dict = {
        'replicate_id': list(range(num_replicates)),
    }
    df_dict.update({
        distribution_name: [
            diff_dict_distribution[replicate_id]
            for replicate_id in range(num_replicates)
        ]
        for distribution_name, diff_dict_distribution in diff_dict.items()
    })
    df = pd.DataFrame(df_dict)
    df.to_csv(
        f"{output_folder}/distribution_distance.csv",
        index=False,
        float_format='%.4f',
    )


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

    # distributions_df_list = \
    #     compare_distributions(
    #         network_1_folder,
    #         network_2_folder,
    #     )

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
    # print(df)
    df.to_csv(output_file, index=False, float_format='%.4f')


if __name__ == "__main__":
    compare_stats()
