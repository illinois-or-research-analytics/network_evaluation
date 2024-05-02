import glob

import click
import networkit as nk
import numpy as np

def parse_stats(stats_path):
    stat_dict = {}
    with open(stats_path, "r") as f:
        for lines in f:
            line_arr = lines.strip().split(",")
            stat_name = line_arr[0]
            stat_type = line_arr[1]
            stat_value = line_arr[2]
            if stat_type == "scalar":
                stat_dict[stat_name] = float(stat_value)
    return stat_dict

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

    input_scalar_stat_dict = parse_stats(f"{input_network_folder}/stats.dat")
    for current_replicate_index,current_replicate_folder in enumerate(replicate_folder_arr):
        replicates_scalar_stat_dict[current_replicate_index] = parse_stats(f"{current_replicate_folder}/stats.dat")

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


@click.command()
@click.option("--input-network-folder", required=True, type=click.Path(exists=True), help="Input network stats folder")
@click.option("--input-replicates-folder", required=True, type=click.Path(exists=True), help="Input replicates stats folder")
@click.option("--output-file", required=True, type=click.Path(), help="Ouput file")
def compare_stats(input_network_folder, input_replicates_folder, output_file):
    """ input network folder needs a stats.dat in there at least
    input replicates folder needs subfolders where each subfolder has a stats.dat at least
    """

    compare_scalars(input_network_folder, input_replicates_folder, output_file)


if __name__ == "__main__":
    compare_stats()
