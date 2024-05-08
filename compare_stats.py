import os
import glob
import json
from pathlib import Path

import click
import scipy
import numpy as np
import pandas as pd

from constants import *


def parse_distribution(distribution_path):
    distribution_arr = []
    with open(distribution_path, "r") as f:
        for line in f:
            distribution_arr.append(float(line))
    return distribution_arr


def parse_json(stats_path):
    with open(stats_path, "r") as f:
        return json.load(f)


def compare_scalars(input_network_folder, input_replicates_folder, output_file):
    input_scalar_stat_dict = {
    }
    replicates_scalar_stat_dict = {
    }
    replicates_scalar_mean_dict = {
    }
    replicates_scalar_stderr_dict = {
    }
    replicate_folder_arr = glob.glob(f"{input_replicates_folder}/*")
    num_replicates = len(replicate_folder_arr)
    for current_replicate_index in range(num_replicates):
        replicates_scalar_stat_dict[current_replicate_index] = {}

    input_scalar_stat_dict = parse_json(f"{input_network_folder}/stats.json")
    for current_replicate_index, current_replicate_folder in enumerate(replicate_folder_arr):
        replicates_scalar_stat_dict[current_replicate_index] = parse_json(
            f"{current_replicate_folder}/stats.json")

    for input_scalar_name, _ in input_scalar_stat_dict.items():
        current_value_arr = []
        for current_replicate_index in range(num_replicates):
            current_value_arr.append(
                replicates_scalar_stat_dict[current_replicate_index][input_scalar_name])
        replicates_scalar_mean_dict[input_scalar_name] = np.mean(
            current_value_arr)
        replicates_scalar_stderr_dict[input_scalar_name] = np.std(
            current_value_arr) / np.sqrt(num_replicates)

    for input_scalar_name, input_scalar_value in input_scalar_stat_dict.items():
        current_mean = replicates_scalar_mean_dict[input_scalar_name]
        current_stderr = replicates_scalar_stderr_dict[input_scalar_name]
        current_relative_difference = (
            current_mean - input_scalar_value) / current_stderr
        current_absolute_difference = (
            current_mean - input_scalar_value) / input_scalar_value
        print(f"{input_scalar_name} relative difference: {
              current_relative_difference}")
        print(f"{input_scalar_name} absolute difference: {
              current_absolute_difference}")


def compare_distributions(input_network_folder, input_replicates_folder, output_file):
    input_distribution_stat_dict = {
    }
    replicates_distribution_stat_dict = {
    }
    replicate_folder_arr = glob.glob(f"{input_replicates_folder}/*")
    num_replicates = len(replicate_folder_arr)

    current_distribution_arr = glob.glob(
        f"{input_network_folder}/*.distribution")
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

    for current_distribution_name in distribution_name_arr:
        print(f"evaluating {current_distribution_name} k-s stat")
        for current_replicate_index in range(num_replicates):
            input_distribution = input_distribution_stat_dict[current_distribution_name]
            replicate_distribution = replicates_distribution_stat_dict[
                current_replicate_index][current_distribution_name]
            try:
                print(f"replicate {current_replicate_index}: {
                      scipy.stats.ks_2samp(input_distribution, replicate_distribution)}")
            except Exception as e:
                print(f"error: {e}")


def compare_sequences(input_network_folder, input_replicates_folder, output_file):
    input_sequence_stat_dict = {
    }
    replicates_sequence_stat_dict = {
    }
    replicate_folder_arr = glob.glob(f"{input_replicates_folder}/*")
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
        }

        replicates_sequence_stat_dict[current_replicate_index] = {}
        for current_sequence_name in sequence_name_arr:
            current_sequence_file = \
                f"{current_replicate_folder}/{current_sequence_name}.distribution"
            replicates_sequence_stat_dict[current_replicate_index][current_sequence_name] = \
                parse_distribution(current_sequence_file)

    for current_sequence_name in sequence_name_arr:
        if current_sequence_name in NODE_DISTR_STATS:
            df_input_ids = input_ids_df_dict['node']
            assert df_input_ids is not None
        elif current_sequence_name in COMM_DISTR_STATS:
            df_input_ids = input_ids_df_dict['comm']
            assert df_input_ids is not None
        else:
            continue

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

            try:
                d = np.linalg.norm(
                    df_joined['input'] - df_joined['replicate'], ord=1)
                print(f"replicate {current_replicate_index}: {d}")
            except Exception as e:
                print(f"error: {e}")


@ click.command()
@ click.option("--input-network-folder", required=True, type=click.Path(exists=True), help="Input network stats folder")
@ click.option("--input-replicates-folder", required=True, type=click.Path(exists=True), help="Input replicates stats folder")
@ click.option("--output-file", required=True, type=click.Path(), help="Ouput file")
def compare_stats(input_network_folder, input_replicates_folder, output_file):
    """ input network folder needs a stats.dat in there at least
    input replicates folder needs subfolders where each subfolder has a stats.dat at least
    """
    # print("comparing scalars")
    # compare_scalars(input_network_folder, input_replicates_folder, output_file)
    # print("comparing distributions")
    # compare_distributions(input_network_folder,
    #                       input_replicates_folder, output_file)
    print("comparing sequences")
    compare_sequences(input_network_folder,
                      input_replicates_folder, output_file)


if __name__ == "__main__":
    compare_stats()
