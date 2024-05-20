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
        d = d/len(input_sequence)
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
    input_network_folder,
    input_replicates_folder,
    output_folder,
) -> None:
    input_scalar_stat_dict = {
    }
    replicates_scalar_stat_dict = {
    }

    replicate_folder_arr = list(filter(
        os.path.isdir,
        glob.glob(f"{input_replicates_folder}/*"),
    ))
    num_replicates = len(replicate_folder_arr)

    for current_replicate_index in range(num_replicates):
        replicates_scalar_stat_dict[current_replicate_index] = {}

    input_scalar_stat_dict = parse_json(f"{input_network_folder}/stats.json")
    for current_replicate_index, current_replicate_folder in enumerate(replicate_folder_arr):
        replicates_scalar_stat_dict[current_replicate_index] = \
            parse_json(f"{current_replicate_folder}/stats.json")

    relative_diff_dict = {}
    statistical_diff_dict = {}
    for name, input_value in input_scalar_stat_dict.items():
        #The-Anh: maybe just compute and store the difference and percentage change of each replicate
        #i.e. (rep_value - input_value) and (rep_value - input_value) / input_value
        
        current_value_arr = [
            replicates_scalar_stat_dict[replicate_index][name]
            for replicate_index in range(num_replicates)
        ]
        mean = np.mean(current_value_arr)
        stderr = np.std(current_value_arr) / np.sqrt(num_replicates)

        relative_difference = (mean - input_value) / input_value
        statistical_difference = (mean - input_value) / stderr

        print(f'{name}: ', end='')
        print(f'(relative) {relative_difference} | ', end='')
        print(f'(statistical) {statistical_difference}')

        relative_diff_dict[name] = relative_difference
        statistical_diff_dict[name] = statistical_difference

    # Write results to csv
    df_dict = {
        'name': list(relative_diff_dict.keys()),
        'relative_diff': list(relative_diff_dict.values()),
        'statistical_diff': list(statistical_diff_dict.values()),
    }
    df = pd.DataFrame(df_dict)
    df.to_csv(
        f"{output_folder}/scalar_distance.csv",
        index=False,
        float_format='%.4f',
    )


def compare_distributions(
    input_folder,
    replicates_folder,
    output_folder,
) -> None:
    input_distribution_stat_dict = {
    }
    replicates_distribution_stat_dict = {
    }

    replicate_folder_arr = list(filter(
        os.path.isdir,
        glob.glob(f"{replicates_folder}/*"),
    ))
    num_replicates = len(replicate_folder_arr)

    current_distribution_arr = glob.glob(
        f"{input_folder}/*.distribution")
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
    input_sequence_stat_dict = {
    }
    replicates_sequence_stat_dict = {
    }

    replicate_folder_arr = list(filter(
        os.path.isdir,
        glob.glob(f"{input_replicates_folder}/*"),
    ))
    num_replicates = len(replicate_folder_arr)

    input_ids_df_dict = {
        'node':
            pd.read_csv(f"{input_network_folder}/{NODE_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{input_network_folder}/{NODE_ORDER_FN}")
            else None,
        'comm':
            pd.read_csv(f"{input_network_folder}/{COMM_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{input_network_folder}/{COMM_ORDER_FN}")
            else None,
        'o_node':
            pd.read_csv(f"{input_network_folder}/{OUTLIER_ORDER_FN}",
                        header=None, names=['id'])
            if os.path.exists(f"{input_network_folder}/{OUTLIER_ORDER_FN}")
            else None,
    }

    current_sequence_arr = glob.glob(
        f"{input_network_folder}/*.distribution")
    sequence_name_arr = []
    for current_sequence_file in current_sequence_arr:
        sequence_name = Path(current_sequence_file).stem
        sequence_name_arr.append(sequence_name)
        input_sequence_stat_dict[sequence_name] = parse_distribution(
            current_sequence_file)

    replicate_ids_df_dict = {}
    for current_replicate_index in range(num_replicates):
        current_replicate_folder = replicate_folder_arr[current_replicate_index]

        replicate_ids_df_dict[current_replicate_index] = {
            'node':
                pd.read_csv(
                    f"{current_replicate_folder}/{NODE_ORDER_FN}",
                    header=None,
                    names=['id'],
                )
                if os.path.exists(f"{current_replicate_folder}/{NODE_ORDER_FN}")
                else None,
            'comm':
                pd.read_csv(
                    f"{current_replicate_folder}/{COMM_ORDER_FN}",
                    header=None,
                    names=['id'],
                )
                if os.path.exists(f"{current_replicate_folder}/{COMM_ORDER_FN}")
                else None,
            'o_node':
                pd.read_csv(
                    f"{current_replicate_folder}/{OUTLIER_ORDER_FN}",
                    header=None,
                    names=['id'],
                )
                if os.path.exists(f"{current_replicate_folder}/{OUTLIER_ORDER_FN}")
                else None,
        }

        replicates_sequence_stat_dict[current_replicate_index] = {}
        for current_sequence_name in sequence_name_arr:
            current_sequence_file = \
                f"{current_replicate_folder}/{current_sequence_name}.distribution"
            replicates_sequence_stat_dict[current_replicate_index][current_sequence_name] = \
                parse_distribution(current_sequence_file)

    diff_dict = dict()
    diff_dict2 = dict()
    for current_sequence_name in sequence_name_arr:
        if current_sequence_name in NODE_DISTR_STATS:
            df_input_ids = input_ids_df_dict['node']
            assert df_input_ids is not None
        elif current_sequence_name in COMM_DISTR_STATS:
            df_input_ids = input_ids_df_dict['comm']
            assert df_input_ids is not None
        elif current_sequence_name in OUTLIER_DISTR_STATS:
            df_input_ids = input_ids_df_dict['o_node']
            assert df_input_ids is not None
        else:
            continue

        diff_dict[current_sequence_name] = dict()
        diff_dict2[current_sequence_name+"_l2"] = dict()

        print(f"evaluating {current_sequence_name}")

        df_input_values = pd.DataFrame(
            input_sequence_stat_dict[current_sequence_name], columns=['input'])

        assert len(df_input_ids) == len(df_input_values)

        df_input = pd.concat([df_input_ids, df_input_values], axis=1)

        for current_replicate_index in range(num_replicates):
            if current_sequence_name in NODE_DISTR_STATS:
                df_replicate_ids = replicate_ids_df_dict[current_replicate_index]['node']
                assert df_replicate_ids is not None
            elif current_sequence_name in COMM_DISTR_STATS:
                df_replicate_ids = replicate_ids_df_dict[current_replicate_index]['comm']
                assert df_replicate_ids is not None
            elif current_sequence_name in OUTLIER_DISTR_STATS:
                df_replicate_ids = replicate_ids_df_dict[current_replicate_index]['o_node']
                assert df_replicate_ids is not None
            else:
                continue

            df_replicate_values = pd.DataFrame(
                replicates_sequence_stat_dict[current_replicate_index][current_sequence_name], columns=['replicate'])

            assert len(df_replicate_ids) == len(df_replicate_values)

            df_replicate = pd.concat(
                [df_replicate_ids, df_replicate_values], axis=1)

            # assert len(df_input) == len(df_replicate), f"{
            #     len(df_input)} != {len(df_replicate)}"

            # TODO: Handle cases where there are outliers. For now, just ignore them.
            df_joined = pd.merge(df_input, df_replicate, on='id', how='inner')

            # assert len(df_joined) == len(df_input) and len(
            #     df_joined) == len(df_replicate)

            diff,diff2 = sequence_distance(
                df_joined['input'].values,
                df_joined['replicate'].values,
            )
            diff_dict[current_sequence_name][current_replicate_index] = diff
            diff_dict2[current_sequence_name+"_l2"][current_replicate_index] = diff2
            print(f"replicate {current_replicate_index}: {diff}")
            print(f"replicate {current_replicate_index}: {diff2}")

    # Write results to csv
    df_dict = {
        'replicate_id': list(range(num_replicates)),
    }
    df_dict.update({
        sequence_name: [
            diff_dict_seq[replicate_id]
            for replicate_id in range(num_replicates)
        ]
        for sequence_name, diff_dict_seq in diff_dict.items()
    })
    df_dict.update({
        sequence_name: [
            diff_dict_seq[replicate_id]
            for replicate_id in range(num_replicates)
        ]
        for sequence_name, diff_dict_seq in diff_dict2.items()
    })
    df = pd.DataFrame(df_dict)
    df.to_csv(
        f"{output_folder}/sequence_distance.csv",
        index=False,
        float_format='%.4f',
    )


@ click.command()
@ click.option("--input-network-folder", required=True, type=click.Path(exists=True), help="Input network stats folder")
@ click.option("--input-replicates-folder", required=True, type=click.Path(exists=True), help="Input replicates stats folder")
@ click.option("--output-folder", required=True, type=click.Path(), help="Ouput folder to save the comparison results")
def compare_stats(input_network_folder, input_replicates_folder, output_folder):
    """ input network folder needs a stats.dat in there at least
    input replicates folder needs subfolders where each subfolder has a stats.dat at least
    """
    os.makedirs(output_folder, exist_ok=True)

    print("comparing scalars")
    compare_scalars(
        input_network_folder,
        input_replicates_folder,
        output_folder,
    )

    print("comparing distributions")
    compare_distributions(
        input_network_folder,
        input_replicates_folder,
        output_folder,
    )

    print("comparing sequences")
    compare_sequences(
        input_network_folder,
        input_replicates_folder,
        output_folder,
    )


if __name__ == "__main__":
    compare_stats()
