import os
import argparse

import numpy as np
import graph_tool.all as gt
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pair_confusion_matrix


NODE_COLUMN_NAMES = [
    "node_id",
    "node",
    "vertex_id",
    "vertex",
    "source_id",
    "source",
    "target_id",
    "target",
    "node1_id",
    "node1",
    "node2_id",
    "node2",
]

COMMUNITY_COLUMN_NAMES = [
    "community_id",
    "community",
    "cluster_id",
    "cluster",
    "com_id",
    "com",
]


def get_delimiter(filepath):
    with open(filepath, "r") as f:
        for line in f:
            current_line = line.strip()
            if current_line[0] == "#":
                continue
            if "," in current_line:
                return ","
            elif " " in current_line:
                return " "
            elif "\t" in current_line:
                return "\t"


def get_node_set_edgelist(edgelist):
    node_set = set()
    current_delimiter = get_delimiter(edgelist)
    with open(edgelist, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            parts = line.strip().split(current_delimiter)
            if (
                len(parts) >= 2
                and parts[0].lower() in NODE_COLUMN_NAMES
                and parts[1].lower() in NODE_COLUMN_NAMES
            ):
                continue
            u, v = parts[0], parts[1]
            node_set.add(u)
            node_set.add(v)
    return node_set


def get_node_set_clustering(filepath):
    non_singleton_node_set = set()
    current_delimiter = get_delimiter(filepath)
    cluster_to_node_id_dict = dict()
    with open(filepath, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            parts = line.strip().split(current_delimiter)
            if (
                len(parts) >= 2
                and parts[0].lower() in NODE_COLUMN_NAMES
                and parts[1].lower() in COMMUNITY_COLUMN_NAMES
            ):
                continue
            node_id, cluster_id = parts[0], parts[1]
            if cluster_id not in cluster_to_node_id_dict:
                cluster_to_node_id_dict[cluster_id] = []
            cluster_to_node_id_dict[cluster_id].append(node_id)
    for cluster_id, cluster_member_arr in cluster_to_node_id_dict.items():
        if len(cluster_member_arr) > 1:
            for cluster_member in cluster_member_arr:
                non_singleton_node_set.add(cluster_member)
    return non_singleton_node_set


def create_mapping(node_set):
    original_to_integer_node_id_dict = dict()
    new_integer_id = 0
    for original_node_id in node_set:
        original_to_integer_node_id_dict[original_node_id] = new_integer_id
        new_integer_id += 1
    return original_to_integer_node_id_dict


def read_clustering(filepath, original_to_integer_node_id_dict):
    current_partition = np.full(
        len(original_to_integer_node_id_dict), "singletonclustersalt"
    )
    current_delimiter = get_delimiter(filepath)
    with open(filepath, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            parts = line.strip().split(current_delimiter)
            if (
                len(parts) >= 2
                and parts[0].lower() in NODE_COLUMN_NAMES
                and parts[1].lower() in COMMUNITY_COLUMN_NAMES
            ):
                continue
            node_id, cluster_id = parts[0], parts[1]
            if node_id in original_to_integer_node_id_dict:
                current_partition[original_to_integer_node_id_dict[node_id]] = (
                    cluster_id
                )

    current_integer_cluster_id = 0
    raw_cluster_to_integer_id_dict = dict()
    for current_integer_node_id in range(len(current_partition)):
        raw_cluster_id = current_partition[current_integer_node_id]
        if raw_cluster_id == "singletonclustersalt":
            current_partition[current_integer_node_id] = current_integer_cluster_id
            current_integer_cluster_id += 1
        else:
            if raw_cluster_id not in raw_cluster_to_integer_id_dict:
                raw_cluster_to_integer_id_dict[raw_cluster_id] = (
                    current_integer_cluster_id
                )
                current_integer_cluster_id += 1
            current_partition[current_integer_node_id] = raw_cluster_to_integer_id_dict[
                raw_cluster_id
            ]
    return current_partition


def get_confusion_matrix(gt_partition, est_partition, matrix=None):
    if matrix is None:
        return pair_confusion_matrix(gt_partition, est_partition)
    return matrix


def calc_precision(matrix):
    twice_tp = matrix[1, 1]
    twice_fp = matrix[0, 1]
    return twice_tp / (twice_tp + twice_fp) if (twice_tp + twice_fp) > 0 else 0.0


def calc_recall(matrix):
    twice_tp = matrix[1, 1]
    twice_fn = matrix[1, 0]
    return twice_tp / (twice_tp + twice_fn) if (twice_tp + twice_fn) > 0 else 0.0


def calc_f1_score(matrix):
    precision = calc_precision(matrix)
    recall = calc_recall(matrix)
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def calc_fnr(matrix):
    twice_fn = matrix[1, 0]
    twice_tp = matrix[1, 1]
    return twice_fn / (twice_fn + twice_tp) if (twice_fn + twice_tp) > 0 else 0.0


def calc_fpr(matrix):
    twice_fp = matrix[0, 1]
    twice_tn = matrix[0, 0]
    return twice_fp / (twice_fp + twice_tn) if (twice_fp + twice_tn) > 0 else 0.0


def clustering_accuracy(
    input_edgelist,
    groundtruth_clustering,
    estimated_clustering,
    output_prefix,
    num_processors=1,
    local=False,
):
    if not os.path.exists(input_edgelist):
        raise FileNotFoundError(f"Input edgelist file {input_edgelist} does not exist.")

    gt.openmp_set_num_threads(num_processors)
    node_set = get_node_set_edgelist(input_edgelist)
    original_node_set_length = len(node_set)
    estimated_node_set = get_node_set_clustering(estimated_clustering)
    if local:
        node_set = estimated_node_set

    original_to_integer_node_id_dict = create_mapping(node_set)
    if len(original_to_integer_node_id_dict) == 0:
        raise ValueError("No nodes found in the specified node set.")

    groundtruth_partition = read_clustering(
        groundtruth_clustering, original_to_integer_node_id_dict
    )
    estimated_partition = read_clustering(
        estimated_clustering, original_to_integer_node_id_dict
    )

    # Write node coverage
    node_coverage_path = output_prefix + ".node_coverage"
    try:
        node_coverage = float(len(estimated_node_set)) / original_node_set_length
        with open(node_coverage_path, "w") as f:
            f.write(f"{node_coverage}\n")
    except Exception as e:
        print(f"Error writing node coverage: {e}")

    # Write NMI
    nmi_path = output_prefix + ".nmi"
    try:
        current_nmi = gt.mutual_information(
            groundtruth_partition, estimated_partition, norm=True, adjusted=False
        )
        with open(nmi_path, "w") as f:
            f.write(f"{current_nmi}\n")
    except Exception as e:
        print(f"Error writing NMI: {e}")

    # Write AMI
    ami_path = output_prefix + ".ami"
    try:
        current_ami = gt.mutual_information(
            groundtruth_partition, estimated_partition, adjusted=True
        )
        with open(ami_path, "w") as f:
            f.write(f"{current_ami}\n")
    except Exception as e:
        print(f"Error writing AMI: {e}")

    # Write ARI
    ari_path = output_prefix + ".ari"
    try:
        current_ari = adjusted_rand_score(groundtruth_partition, estimated_partition)
        with open(ari_path, "w") as f:
            f.write(f"{current_ari}\n")
    except Exception as e:
        print(f"Error writing ARI: {e}")

    # Write FNR, FPR, Precision, Recall, F1-score (metrics based on pair confusion matrix)
    # to avoid redundant computation, we compute the pair confusion matrix once
    # also note that we consider pairs of nodes
    # so TP, FP, FN, TN are all doubled compared to the standard definition,
    # but this does not affect the final metric values since they are ratios
    # see more details in the documentation of sklearn.metrics.pair_confusion_matrix
    confusion_matrix = None

    fnr_path = output_prefix + ".fnr"
    try:
        confusion_matrix = get_confusion_matrix(
            groundtruth_partition, estimated_partition, confusion_matrix
        )
        current_fnr = calc_fnr(confusion_matrix)
        with open(fnr_path, "w") as f:
            f.write(f"{current_fnr}\n")
    except Exception as e:
        print(f"Error writing FNR: {e}")

    fpr_path = output_prefix + ".fpr"
    try:
        confusion_matrix = get_confusion_matrix(
            groundtruth_partition, estimated_partition, confusion_matrix
        )
        current_fpr = calc_fpr(confusion_matrix)
        with open(fpr_path, "w") as f:
            f.write(f"{current_fpr}\n")
    except Exception as e:
        print(f"Error writing FPR: {e}")

    precision_path = output_prefix + ".precision"
    try:
        confusion_matrix = get_confusion_matrix(
            groundtruth_partition, estimated_partition, confusion_matrix
        )
        current_precision = calc_precision(confusion_matrix)
        with open(precision_path, "w") as f:
            f.write(f"{current_precision}\n")
    except Exception as e:
        print(f"Error writing Precision: {e}")

    recall_path = output_prefix + ".recall"
    try:
        confusion_matrix = get_confusion_matrix(
            groundtruth_partition, estimated_partition, confusion_matrix
        )
        current_recall = calc_recall(confusion_matrix)
        with open(recall_path, "w") as f:
            f.write(f"{current_recall}\n")
    except Exception as e:
        print(f"Error writing Recall: {e}")

    f1_score_path = output_prefix + ".f1_score"
    try:
        confusion_matrix = get_confusion_matrix(
            groundtruth_partition, estimated_partition, confusion_matrix
        )
        current_f1_score = calc_f1_score(confusion_matrix)
        with open(f1_score_path, "w") as f:
            f.write(f"{current_f1_score}\n")
    except Exception as e:
        print(f"Error writing F1-score: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute clustering accuracy metrics.")
    parser.add_argument("--input-network", help="Path to the input edgelist file")
    parser.add_argument(
        "--gt-clustering", help="Path to the groundtruth clustering file"
    )
    parser.add_argument(
        "--est-clustering", help="Path to the estimated clustering file"
    )
    parser.add_argument(
        "--output-prefix", help="Path to the output prefix (no extension)"
    )
    parser.add_argument(
        "--num_processors",
        type=int,
        default=1,
        help="Number of processors to use (default: 1)",
    )
    parser.add_argument(
        "--local", action="store_true", help="Use only nodes in estimated clustering"
    )

    args = parser.parse_args()

    clustering_accuracy(
        args.input_network,
        args.gt_clustering,
        args.est_clustering,
        args.output_prefix,
        num_processors=args.num_processors,
        local=args.local,
    )
