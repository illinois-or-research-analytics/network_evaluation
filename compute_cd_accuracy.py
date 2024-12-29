import numpy as np
import click
import graph_tool.all as gt
from sklearn.metrics import adjusted_rand_score

import multiprocessing as mp


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
            u, v = line.strip().split(current_delimiter)
            node_set.add(u)
            node_set.add(v)
    return node_set


def create_mapping(node_set):
    original_to_integer_node_id_dict = dict()
    new_integer_id = 0
    for original_node_id in node_set:
        original_to_integer_node_id_dict[original_node_id] = new_integer_id
        new_integer_id += 1
    return original_to_integer_node_id_dict


def get_node_set_clustering(filepath):
    non_singleton_node_set = set()
    current_delimiter = get_delimiter(filepath)
    cluster_to_node_id_dict = dict()
    with open(filepath, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            node_id, cluster_id = line.strip().split(current_delimiter)
            if cluster_id not in cluster_to_node_id_dict:
                cluster_to_node_id_dict[cluster_id] = []
            cluster_to_node_id_dict[cluster_id].append(node_id)
    for cluster_id, cluster_member_arr in cluster_to_node_id_dict.items():
        if len(cluster_member_arr) > 1:
            for cluster_member in cluster_member_arr:
                non_singleton_node_set.add(cluster_member)
    return non_singleton_node_set


def read_clustering(filepath, original_to_integer_node_id_dict):
    current_partition = np.full(
        len(original_to_integer_node_id_dict), "singletonclustersalt")
    current_delimiter = get_delimiter(filepath)
    with open(filepath, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            node_id, cluster_id = line.strip().split(current_delimiter)
            if node_id in original_to_integer_node_id_dict:
                current_partition[original_to_integer_node_id_dict[node_id]] = cluster_id

    current_integer_cluster_id = 0
    raw_cluster_to_integer_id_dict = dict()
    for current_integer_node_id in range(len(current_partition)):
        raw_cluster_id = current_partition[current_integer_node_id]
        if raw_cluster_id == "singletonclustersalt":
            current_partition[current_integer_node_id] = current_integer_cluster_id
            current_integer_cluster_id += 1
        else:
            if raw_cluster_id not in raw_cluster_to_integer_id_dict:
                raw_cluster_to_integer_id_dict[raw_cluster_id] = current_integer_cluster_id
                current_integer_cluster_id += 1
            current_partition[current_integer_node_id] = raw_cluster_to_integer_id_dict[raw_cluster_id]
    return current_partition


def get_edge_mask(edgelist, partition, original_to_integer_node_id_dict):
    # https://codeocean.com/capsule/0712485/tree/v1
    edge_mask = []
    current_delimiter = get_delimiter(edgelist)
    with open(edgelist, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            u, v = line.strip().split(current_delimiter)
            u_cluster = partition[original_to_integer_node_id_dict[u]]
            v_cluster = partition[original_to_integer_node_id_dict[v]]
            edge_mask.append(u_cluster == v_cluster)
    return np.array(edge_mask)


def get_agri(groundtruth_partition, estimated_partition, edgelist, original_to_integer_node_id_dict):
    # https://codeocean.com/capsule/0712485/tree/v1
    groundtruth_edgemask = get_edge_mask(
        edgelist, groundtruth_partition, original_to_integer_node_id_dict)
    estimated_edgemask = get_edge_mask(
        edgelist, estimated_partition, original_to_integer_node_id_dict)
    gt_sum = sum(groundtruth_edgemask)
    estimated_sum = sum(estimated_edgemask)
    both_sum = sum(groundtruth_edgemask * estimated_edgemask)
    size = len(groundtruth_edgemask)
    return (both_sum - gt_sum * estimated_sum / size) / (0.5 * (gt_sum + estimated_sum) - gt_sum * estimated_sum / size)


def get_cluster_node_pairs(partition):
    n = len(partition)
    return set([(i, j) for i in range(n) for j in range(i + 1, n) if partition[i] == partition[j]])


def get_fnr_fpr(groundtruth_partition, estimated_partition, num_processors):
    with mp.Pool(processes=num_processors) as pool:
        results = pool.map(get_cluster_node_pairs, [
                           groundtruth_partition, estimated_partition])
    tp = len(results[0].intersection(results[1]))
    fn = len(results[0]) - tp
    fp = len(results[1]) - tp
    n = len(groundtruth_partition)
    tn = int((n * (n - 1)) / 2) - len(results[1]) - fn
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    return {
        "fnr": fnr,
        "fpr": fpr,
    }


def min_accuracy(input_edgelist, groundtruth_clustering, estimated_clustering, output_file, num_processors=1, local=False):
    with open(output_file, "w") as f:
        f.write(f"starting {groundtruth_clustering} vs {
                estimated_clustering} on {input_edgelist}\n")

    gt.openmp_set_num_threads(num_processors)
    node_set = get_node_set_edgelist(input_edgelist)
    original_node_set_length = len(node_set)
    estimated_node_set = get_node_set_clustering(estimated_clustering)
    if local:
        node_set = estimated_node_set

    original_to_integer_node_id_dict = create_mapping(node_set)
    if len(original_to_integer_node_id_dict) == 0:
        with open(output_file, "a") as f:
            f.write(f"empty or only singleton nodes in estimated clustering file")
            return

    groundtruth_partition = read_clustering(
        groundtruth_clustering, original_to_integer_node_id_dict)
    estimated_partition = read_clustering(
        estimated_clustering, original_to_integer_node_id_dict)
    # fnr_fpr_dict = get_fnr_fpr(groundtruth_partition, estimated_partition, num_processors)
    # with open(output_file, "a") as f:
    # f.write(f"fnr:{fnr_fpr_dict['fnr']}\n")
    # f.write(f"fpr:{fnr_fpr_dict['fpr']}\n")

    with open(output_file, "a") as f:
        f.write(f"node coverage:{
                float(len(estimated_node_set)) / original_node_set_length}\n")

    current_nmi = gt.mutual_information(
        groundtruth_partition, estimated_partition, norm=True, adjusted=False)
    with open(output_file, "a") as f:
        f.write(f"nmi:{current_nmi}\n")
    # current_ami = gt.mutual_information(
    #     groundtruth_partition, estimated_partition, adjusted=True)
    # with open(output_file, "a") as f:
    #     f.write(f"ami:{current_ami}\n")
    # current_rnmi = gt.reduced_mutual_information(groundtruth_partition, estimated_partition, norm=True)
    # with open(output_file, "a") as f:
    #     f.write(f"rnmi:{current_rnmi}\n")
    # current_nvi = gt.variation_information(groundtruth_partition, estimated_partition, norm=True)
    # with open(output_file, "a") as f:
    #     f.write(f"nvi:{current_nvi}\n")
    current_ari = adjusted_rand_score(
        groundtruth_partition, estimated_partition)
    with open(output_file, "a") as f:
        f.write(f"ari:{current_ari}\n")
    # current_agri = get_agri(groundtruth_partition, estimated_partition, input_edgelist, original_to_integer_node_id_dict)
    # with open(output_file, "a") as f:
    #     f.write(f"agri:{current_agri}\n")


if __name__ == '__main__':
    min_accuracy()
