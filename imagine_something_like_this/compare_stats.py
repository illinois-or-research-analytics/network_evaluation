import glob
import json
from pathlib import Path

import click
import networkit as nk
import numpy as np
import scipy

def parse_sequence(sequence_path):
    sequence_arr = []
    with open(sequence_path, "r") as f:
        for line in f:
            sequence_arr.append(float(line))
    return sequence_arr

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
    for current_replicate_index,current_replicate_folder in enumerate(replicate_folder_arr):
        replicates_scalar_stat_dict[current_replicate_index] = parse_json(f"{current_replicate_folder}/stats.json")

    for input_scalar_name,_ in input_scalar_stat_dict.items():
        current_value_arr = []
        for current_replicate_index in range(num_replicates):
            current_value_arr.append(replicates_scalar_stat_dict[current_replicate_index][input_scalar_name])
        replicates_scalar_mean_dict[input_scalar_name] = np.mean(current_value_arr)
        replicates_scalar_stderr_dict[input_scalar_name] = np.std(current_value_arr) / np.sqrt(num_replicates)

    for input_scalar_name,input_scalar_value in input_scalar_stat_dict.items():
        current_mean = replicates_scalar_mean_dict[input_scalar_name]
        current_stderr = replicates_scalar_stderr_dict[input_scalar_name]
        current_relative_difference = (current_mean - input_scalar_value) / current_stderr
        current_absolute_difference = (current_mean - input_scalar_value) / input_scalar_value
        print(f"{input_scalar_name} relative difference: {current_relative_difference}")
        print(f"{input_scalar_name} absolute difference: {current_absolute_difference}")



def compare_distributions(input_network_folder, input_replicates_folder, output_file):
    input_distribution_stat_dict = {
    }
    replicates_distribution_stat_dict = {
    }
    replicate_folder_arr = glob.glob(f"{input_replicates_folder}/*")
    num_replicates = len(replicate_folder_arr)

    current_distribution_arr = glob.glob(f"{input_network_folder}/*.distribution")
    distribution_name_arr = []
    for current_distribution_file in current_distribution_arr:
        distribution_name = Path(current_distribution_file).stem
        distribution_name_arr.append(distribution_name)
        input_distribution_stat_dict[distribution_name] = parse_distribution(current_distribution_file)

    for current_replicate_index in range(num_replicates):
        replicates_distribution_stat_dict[current_replicate_index] = {}
        for current_distribution_name in distribution_name_arr:
            current_distribution_file = f"{replicate_folder_arr[current_replicate_index]}/{current_distribution_name}.distribution"
            replicates_distribution_stat_dict[current_replicate_index][current_distribution_name] = parse_distribution(current_distribution_file)


    for current_distribution_name in distribution_name_arr:
        print(f"evaluating {current_distribution_name} k-s stat")
        for current_replicate_index in range(num_replicates):
            input_distribution = input_distribution_stat_dict[current_distribution_name]
            replicate_distribution = replicates_distribution_stat_dict[current_replicate_index][current_distribution_name]
            print(f"replicate {current_replicate_index}: {scipy.stats.ks_2samp(input_distribution, replicate_distribution)}")

def compare_sequences(input_network_folder, input_replicates_folder, output_file):
    input_sequence_stat_dict = {
    }
    replicates_sequence_stat_dict = {
    }
    replicate_folder_arr = glob.glob(f"{input_replicates_folder}/*")
    num_replicates = len(replicate_folder_arr)

    current_sequence_arr = glob.glob(f"{input_network_folder}/*.sequence")
    sequence_name_arr = []
    for current_sequence_file in current_sequence_arr:
        sequence_name = Path(current_sequence_file).stem
        sequence_name_arr.append(sequence_name)
        input_sequence_stat_dict[sequence_name] = parse_sequence(current_sequence_file)

    for current_replicate_index in range(num_replicates):
        replicates_sequence_stat_dict[current_replicate_index] = {}
        for current_sequence_name in sequence_name_arr:
            current_sequence_file = f"{replicate_folder_arr[current_replicate_index]}/{current_sequence_name}.sequence"
            replicates_sequence_stat_dict[current_replicate_index][current_sequence_name] = parse_sequence(current_sequence_file)


    for current_sequence_name in sequence_name_arr:
        print(f"evaluating {current_sequence_name} l2 norm")
        for current_replicate_index in range(num_replicates):
            input_sequence = input_sequence_stat_dict[current_sequence_name]
            replicate_sequence = replicates_sequence_stat_dict[current_replicate_index][current_sequence_name]
            difference_sequence = np.subtract(input_sequence, replicate_sequence)
            print(f"replicate {current_replicate_index}: {np.linalg.norm(difference_sequence, ord=2)}")

@click.command()
@click.option("--input-network-folder", required=True, type=click.Path(exists=True), help="Input network stats folder")
@click.option("--input-replicates-folder", required=True, type=click.Path(exists=True), help="Input replicates stats folder")
@click.option("--output-file", required=True, type=click.Path(), help="Ouput file")
def compare_stats(input_network_folder, input_replicates_folder, output_file):
    """ input network folder needs a stats.dat in there at least
    input replicates folder needs subfolders where each subfolder has a stats.dat at least
    """
    print("comparing scalars")
    compare_scalars(input_network_folder, input_replicates_folder, output_file)
    print("comparing distributions")
    compare_distributions(input_network_folder, input_replicates_folder, output_file)
    print("comparing sequences")
    compare_sequences(input_network_folder, input_replicates_folder, output_file)


if __name__ == "__main__":
    compare_stats()
