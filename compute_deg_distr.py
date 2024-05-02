import os
import csv
import argparse
from typing import Dict

import pandas as pd

from utils import parse_mapping
from constants import (
    EDGE_FILENAME,
    DEG_MAPPING_FILENAME,
    DEG_COLNAME,
    DEG_DISTR_FILENAME,
)


def parse_args(
) -> argparse.Namespace:
    parser: argparse.ArgumentParser = \
        argparse.ArgumentParser(
            description='Compute degree distribution',
        )

    parser.add_argument(
        '-i',
        '--network-dir',
        type=str,
        required=True,
        help='The directory containing the network data',
    )
    return parser.parse_args()


def compute_node_to_degree_map(
    edge_fn: str,
    mapping: Dict[str, int],
) -> pd.DataFrame:
    f = open(edge_fn)
    csv_reader = csv.reader(f, delimiter='\t')

    node2deg = dict()
    for u_, v_ in csv_reader:
        u = mapping.get(u_, int(u_))
        v = mapping.get(v_, int(v_))

        node2deg.setdefault(u, 0)
        node2deg[u] += 1

        node2deg.setdefault(v, 0)
        node2deg[v] += 1

    f.close()

    return node2deg


def compute_deg_distr(
    network_dir: str,
    save_file: bool = True,
) -> pd.DataFrame:
    mapping = parse_mapping(f'{network_dir}/{NODE_MAPPING_FILENAME}')

    node2deg = compute_node_to_degree_map(
        f'{network_dir}/{EDGE_FILENAME}',
        mapping,
    )

    degrees = [
        node2deg[i]
        for i in range(len(node2deg))
    ]
    df = pd.DataFrame(degrees, columns=[DEG_COLNAME])

    if save_file:
        df.to_csv(
            f'{network_dir}/{DEG_DISTR_FILENAME}',
            sep='\t',
            index=False,
            header=False,
        )

    return df


def main(
) -> None:
    args = parse_args()
    df = compute_deg_distr(args.network_dir)
    print(df)


if __name__ == '__main__':
    main()
