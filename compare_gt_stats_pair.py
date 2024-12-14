import os
import glob
import json
from pathlib import Path
from typing import List, Any

import click
import scipy
import numpy as np
import pandas as pd


def parse_json(stats_path):
    with open(stats_path, "r") as f:
        return json.load(f)


def scalar_distance(x, x_bar, dist_type):
    if dist_type == 'abs_diff':
        d = x - x_bar
    elif dist_type == 'rel_diff':
        d = (x - x_bar) / x
    elif dist_type == 'rpd':
        d = (x - x_bar) / (abs(x) + abs(x_bar))
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")
    return d


def compare_scalars(
    network_1_folder,
    network_2_folder,
) -> List[List[Any]]:
    network_1_stats = parse_json(f"{network_1_folder}/gt_stats.json")
    network_2_stats = parse_json(f"{network_2_folder}/gt_stats.json")

    common_stats = set(
        network_1_stats.keys()
    ).intersection(
        network_2_stats.keys()
    )
    common_stats = sorted(common_stats)

    diff_dict = dict()
    for name in common_stats:
        diff_dict[name] = dict()

        for dist_type in ['abs_diff', 'rel_diff', 'rpd']:
            try:
                diff = scalar_distance(
                    network_1_stats[name],
                    network_2_stats[name],
                    dist_type,
                )
            except Exception as e:
                print(f"[ERROR] ({dist_type}) ({name}) {e}")
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
def compare_stats(
    network_1_folder,
    network_2_folder,
    output_file,
) -> None:
    df_list = []

    scalars_df_list = \
        compare_scalars(
            network_1_folder,
            network_2_folder,
        )
    df_list.extend(scalars_df_list)

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
