import os
import json
import argparse
from pathlib import Path
from typing import List, Any

import scipy.stats
import numpy as np
import pandas as pd

# --- Constants for Cluster-Dependent Stats ---
CLUSTER_NODE_ORDER_FN = "node.idx"
CLUSTER_COMM_ORDER_FN = "com.idx"

CLUSTER_SCALAR_STATS = [
    "node_coverage",
    "n_outliers",
    "n_clusters",
    "n_disconnected_clusters",
    "n_connected_clusters",
    "n_wellconnected_clusters",
]

CLUSTER_NODE_DISTR_STATS = ["mixing_parameter"]
CLUSTER_COMM_DISTR_STATS = [
    "m",
    "n",
    "c",
    "conductance",
    "degree_density",
    "edge_density",
    "mincut",
    "modularity",
]

# --- Constants for Network-Only Stats ---
NETWORK_NODE_ORDER_FN = "node.idx"

NETWORK_SCALAR_STATS = [
    "n_nodes",
    "n_edges",
    "n_concomp",
    "mean_degree",
    "deg_assort",
    "mean_kcore",
    "global_ccoeff",
    "local_ccoeff",
    "pseudo_diameter",
    "char_time",
    "node_percolation_targeted",
    "node_percolation_random",
    "frac_giant_ccomp",
]

NETWORK_NODE_DISTR_STATS = [
    "degree",
    "local_ccoeff_nodes",
    "pagerank",
    "betweenness",
    "kcore",
]
NETWORK_GENERAL_DISTR_STATS = ["concomp_sizes"]


# ==========================================
# Math & Distance Helpers
# ==========================================


def scalar_distance(x, x_bar, dist_type):
    if dist_type == "abs_diff":
        return x - x_bar
    elif dist_type == "rel_diff":
        return (x - x_bar) / x if x != 0 else np.nan
    elif dist_type == "rpd":
        return (x - x_bar) / (abs(x) + abs(x_bar)) if (x != 0 or x_bar != 0) else 0.0
    raise ValueError(f"Unknown distance type: {dist_type}")


def distribution_distance(input_dist, rep_dist, dist_type):
    if dist_type == "ks":
        return scipy.stats.ks_2samp(input_dist, rep_dist).statistic
    elif dist_type == "emd":
        return scipy.stats.wasserstein_distance(input_dist, rep_dist)
    raise ValueError(f"Unknown distance type: {dist_type}")


def sequence_distance(seq_1, seq_2, dist_type):
    if dist_type == "l1":
        return np.linalg.norm(seq_1 - seq_2, ord=1)
    elif dist_type == "mean_l1":
        return (
            np.linalg.norm(seq_1 - seq_2, ord=1) / len(seq_1)
            if len(seq_1) > 0
            else np.nan
        )
    elif dist_type == "l2":
        return np.linalg.norm(seq_1 - seq_2, ord=2)
    elif dist_type == "rmse":
        return np.sqrt(np.mean((seq_1 - seq_2) ** 2)) if len(seq_1) > 0 else np.nan
    elif dist_type == "cosine":
        norm_1, norm_2 = np.linalg.norm(seq_1), np.linalg.norm(seq_2)
        return (
            1 - np.dot(seq_1, seq_2) / (norm_1 * norm_2)
            if (norm_1 != 0 and norm_2 != 0)
            else np.nan
        )
    raise ValueError(f"Unknown distance type: {dist_type}")


# ==========================================
# Loaders: Scalars
# ==========================================


def load_cluster_scalars(folder) -> dict:
    res = {}
    if not folder or not os.path.isdir(folder):
        return res
    for name in CLUSTER_SCALAR_STATS:
        path = os.path.join(folder, f"{name}.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                res[name] = float(f.read().strip())
    return res


def load_network_scalars(folder) -> dict:
    res = {}
    if not folder or not os.path.isdir(folder):
        return res
    path = os.path.join(folder, "stats.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        for name in NETWORK_SCALAR_STATS:
            if name in data:
                res[name] = data[name]
    return res


# ==========================================
# Loaders: Distributions
# ==========================================


def load_cluster_distributions(folder) -> dict:
    res = {}
    if not folder or not os.path.isdir(folder):
        return res
    for name in CLUSTER_NODE_DISTR_STATS + CLUSTER_COMM_DISTR_STATS:
        path = os.path.join(folder, f"{name}.txt")
        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            res[name] = df[0].values
    return res


def load_network_distributions(folder) -> dict:
    res = {}
    if not folder or not os.path.isdir(folder):
        return res
    for name in NETWORK_NODE_DISTR_STATS + NETWORK_GENERAL_DISTR_STATS:
        path = os.path.join(folder, f"{name}.distribution")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", header=None)
            res[name] = df[0].values
    return res


# ==========================================
# Loaders: Sequences (Returns DataFrames: id, val)
# ==========================================


def load_cluster_sequences(folder) -> dict:
    res = {}
    if not folder or not os.path.isdir(folder):
        return res

    # 1. Map Node Distributions
    np_path = os.path.join(folder, CLUSTER_NODE_ORDER_FN)
    if os.path.exists(np_path):
        node_ids = pd.read_csv(np_path, header=None, names=["id"], dtype=str)
        for name in CLUSTER_NODE_DISTR_STATS:
            path = os.path.join(folder, f"{name}.txt")
            if os.path.exists(path):
                df_vals = pd.read_csv(path, header=None, names=["val"])
                if len(node_ids) == len(df_vals):
                    res[name] = pd.concat([node_ids, df_vals], axis=1)

    # 2. Map Community Distributions
    cp_path = os.path.join(folder, CLUSTER_COMM_ORDER_FN)
    if os.path.exists(cp_path):
        comm_ids = pd.read_csv(cp_path, header=None, names=["id"], dtype=str)
        for name in CLUSTER_COMM_DISTR_STATS:
            path = os.path.join(folder, f"{name}.txt")
            if os.path.exists(path):
                df_vals = pd.read_csv(path, header=None, names=["val"])
                if len(comm_ids) == len(df_vals):
                    res[name] = pd.concat([comm_ids, df_vals], axis=1)

    return res


def load_network_sequences(folder) -> dict:
    res = {}
    if not folder or not os.path.isdir(folder):
        return res

    idx_path = os.path.join(folder, NETWORK_NODE_ORDER_FN)
    if not os.path.exists(idx_path):
        return res

    node_ids = pd.read_csv(idx_path, header=None, names=["id"], dtype=str)

    # Strictly iterate ONLY over distributions known to map to nodes
    for name in NETWORK_NODE_DISTR_STATS:
        dist_path = os.path.join(folder, f"{name}.distribution")
        if os.path.exists(dist_path):
            df_vals = pd.read_csv(dist_path, sep="\t", header=None, names=["val"])
            if len(node_ids) == len(df_vals):
                res[name] = pd.concat([node_ids, df_vals], axis=1)

    return res


# ==========================================
# Unified Comparators
# ==========================================


def compare_scalars(dict1: dict, dict2: dict, dist_types: List[str]) -> List[List[Any]]:
    df_list = []
    common_stats = set(dict1.keys()).intersection(dict2.keys())
    for name in common_stats:
        val1, val2 = dict1[name], dict2[name]
        for d_type in dist_types:
            try:
                diff = scalar_distance(val1, val2, d_type)
            except Exception:
                diff = np.nan
            df_list.append([name, "scalar", d_type, diff])
    return df_list


def compare_distributions(
    dict1: dict, dict2: dict, dist_types: List[str]
) -> List[List[Any]]:
    df_list = []
    common_stats = set(dict1.keys()).intersection(dict2.keys())
    for name in common_stats:
        d1, d2 = dict1[name], dict2[name]
        if len(d1) == 0 or len(d2) == 0:
            continue
        for d_type in dist_types:
            try:
                diff = distribution_distance(d1, d2, d_type)
            except Exception:
                diff = np.nan
            df_list.append([name, "distribution", d_type, diff])
    return df_list


def compare_sequences(
    dict1: dict, dict2: dict, dist_types: List[str]
) -> List[List[Any]]:
    df_list = []
    common_stats = set(dict1.keys()).intersection(dict2.keys())
    for name in common_stats:
        df1, df2 = dict1[name], dict2[name]
        df_joined = pd.merge(
            df1.rename(columns={"val": "1"}),
            df2.rename(columns={"val": "2"}),
            on="id",
            how="inner",
        )
        if df_joined.empty:
            continue

        seq1, seq2 = df_joined["1"].values, df_joined["2"].values
        for d_type in dist_types:
            try:
                diff = sequence_distance(seq1, seq2, d_type)
            except Exception:
                diff = np.nan
            df_list.append([name, "sequence", d_type, diff])
    return df_list


# ==========================================
# Main Execution
# ==========================================


def parse_args():
    parser = argparse.ArgumentParser(description="Compare network statistics.")

    # Input folders
    parser.add_argument(
        "--cluster-1-folder",
        type=str,
        help="1st cluster stats folder (e.g., Synthetic)",
    )
    parser.add_argument(
        "--cluster-2-folder",
        type=str,
        help="2nd cluster stats folder (e.g., Reference)",
    )
    parser.add_argument(
        "--network-1-folder",
        type=str,
        help="1st network-only stats folder (e.g., Synthetic)",
    )
    parser.add_argument(
        "--network-2-folder",
        type=str,
        help="2nd network-only stats folder (e.g., Reference)",
    )

    # Output file
    parser.add_argument(
        "--output-file", required=True, type=str, help="Output CSV file path"
    )

    # Flags and Distance type overrides
    parser.add_argument(
        "--is-compare-sequence",
        action="store_true",
        help="Compare sequences matching on ID",
    )
    parser.add_argument(
        "--scalar-distances",
        nargs="+",
        default=["abs_diff", "rel_diff", "rpd"],
        help="List of scalar distance metrics to compute",
    )
    parser.add_argument(
        "--dist-distances",
        nargs="+",
        default=["ks", "emd"],
        help="List of distribution distance metrics to compute",
    )
    parser.add_argument(
        "--seq-distances",
        nargs="+",
        default=["l1", "mean_l1", "l2", "rmse", "cosine"],
        help="List of sequence distance metrics to compute",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load & Merge Target 1 (e.g. Synthetic)
    scalars_1 = {
        **load_cluster_scalars(args.cluster_1_folder),
        **load_network_scalars(args.network_1_folder),
    }
    distr_1 = {
        **load_cluster_distributions(args.cluster_1_folder),
        **load_network_distributions(args.network_1_folder),
    }
    seqs_1 = {
        **load_cluster_sequences(args.cluster_1_folder),
        **load_network_sequences(args.network_1_folder),
    }

    # 2. Load & Merge Target 2 (e.g. Reference)
    scalars_2 = {
        **load_cluster_scalars(args.cluster_2_folder),
        **load_network_scalars(args.network_2_folder),
    }
    distr_2 = {
        **load_cluster_distributions(args.cluster_2_folder),
        **load_network_distributions(args.network_2_folder),
    }
    seqs_2 = {
        **load_cluster_sequences(args.cluster_2_folder),
        **load_network_sequences(args.network_2_folder),
    }

    # 3. Compare
    df_list = []

    if scalars_1 and scalars_2:
        print(f"[INFO] Comparing scalars using metrics: {args.scalar_distances}")
        df_list.extend(compare_scalars(scalars_1, scalars_2, args.scalar_distances))

    if distr_1 and distr_2:
        print(f"[INFO] Comparing distributions using metrics: {args.dist_distances}")
        df_list.extend(compare_distributions(distr_1, distr_2, args.dist_distances))

    if args.is_compare_sequence and seqs_1 and seqs_2:
        print(f"[INFO] Comparing sequences using metrics: {args.seq_distances}")
        df_list.extend(compare_sequences(seqs_1, seqs_2, args.seq_distances))

    # 4. Export
    if not df_list:
        print("[WARNING] No comparisons were performed. Exiting.")
        return

    df = pd.DataFrame(
        data=df_list, columns=["stat", "stat_type", "distance_type", "distance"]
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Comparison complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
