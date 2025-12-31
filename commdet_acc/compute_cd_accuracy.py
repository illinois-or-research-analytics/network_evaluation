import numpy as np
import graph_tool.all as gt
from sklearn.metrics import adjusted_rand_score

import multiprocessing as mp
import argparse

import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.metrics import pair_confusion_matrix
import os


def _compute_I_Hg_Hc(contingency_table):
    n = np.sum(contingency_table)
    ng = np.sum(contingency_table, axis=1)
    nc = np.sum(contingency_table, axis=0)
    I = (
        gammaln(n + 1)
        - np.sum(gammaln(ng + 1))
        - np.sum(gammaln(nc + 1))
        + np.sum(gammaln(contingency_table.flatten() + 1))
    )
    Hg = gammaln(n + 1) - np.sum(gammaln(ng + 1))
    Hc = gammaln(n + 1) - np.sum(gammaln(nc + 1))
    I /= np.log(2)
    Hg /= np.log(2)
    Hc /= np.log(2)
    return I, Hg, Hc


def _log_binom(a, b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def _log_Omega_EC(rs, cs, useShortDimension=False, symmetrize=False):
    rs = np.array(rs)
    cs = np.array(cs)
    rs = rs[rs > 0]
    cs = cs[cs > 0]
    if len(rs) == 0 or len(cs) == 0:
        return -np.inf
    if useShortDimension:
        if len(rs) >= len(cs):
            return _log_Omega_EC(rs, cs, useShortDimension=False)
        else:
            return _log_Omega_EC(cs, rs, useShortDimension=False)
    else:
        if symmetrize:
            return (
                _log_Omega_EC(rs, cs, symmetrize=False)
                + _log_Omega_EC(cs, rs, symmetrize=False)
            ) / 2
        else:
            m = len(rs)
            N = sum(rs)
            if N == len(cs):
                return gammaln(N + 1) - sum(gammaln(rs + 1))
            alphaC = (N**2 - N + (N**2 - sum(cs**2)) / m) / (sum(cs**2) - N)
            result = -_log_binom(N + m * alphaC - 1, m * alphaC - 1)
            for r in rs:
                result += _log_binom(r + alphaC - 1, alphaC - 1)
            for c in cs:
                result += _log_binom(c + m - 1, m - 1)
            return result


def _compute_flat_subleading_terms(contingency_table):
    n = np.sum(contingency_table)
    ng = np.sum(contingency_table, axis=1)
    nc = np.sum(contingency_table, axis=0)
    qg = len(ng)
    qc = len(nc)
    delta_Hg = np.log(n) + _log_binom(n - 1, qg - 1)
    delta_Hc = np.log(n) + _log_binom(n - 1, qc - 1)
    delta_HgGc = np.log(n) + _log_binom(n - 1, qg - 1) + _log_Omega_EC(ng, nc)
    delta_HcGg = np.log(n) + _log_binom(n - 1, qc - 1) + _log_Omega_EC(nc, ng)
    delta_HgGg = np.log(n) + _log_binom(n - 1, qg - 1) + _log_Omega_EC(ng, ng)
    delta_HcGc = np.log(n) + _log_binom(n - 1, qc - 1) + _log_Omega_EC(nc, nc)
    delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = (
        delta_Hg / np.log(2),
        delta_Hc / np.log(2),
        delta_HgGc / np.log(2),
        delta_HcGg / np.log(2),
        delta_HgGg / np.log(2),
        delta_HcGc / np.log(2),
    )
    return delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc


def _H_ng_G_alpha(ng, alpha):
    n = np.sum(ng)
    q = len(ng)
    H_ng = _log_binom(n + q * alpha - 1, q * alpha - 1)
    for r in range(q):
        H_ng -= _log_binom(ng[r] + alpha - 1, alpha - 1)
    return H_ng


def _H_ngc_G_nc_alpha(ngc, alpha):
    qg = ngc.shape[0]
    qc = ngc.shape[1]
    nc = np.sum(ngc, axis=0)
    H_ngc = 0
    for s in range(qc):
        H_ngc += _log_binom(nc[s] + qg * alpha - 1, qg * alpha - 1)
        for r in range(qg):
            H_ngc -= _log_binom(ngc[r, s] + alpha - 1, alpha - 1)
    return H_ngc


def _compute_DM_subleading_terms(contingency_table, verbose=False):
    n = np.sum(contingency_table)
    ng = np.sum(contingency_table, axis=1)
    nc = np.sum(contingency_table, axis=0)
    qg = len(ng)
    qc = len(nc)
    ngg = np.diag(ng)
    ncc = np.diag(nc)
    min_alpha = 0.0001
    max_alpha = 10000

    def f(alpha):
        return _H_ng_G_alpha(ng, alpha)

    H_ng_G_alpha = minimize_scalar(f, bounds=(min_alpha, max_alpha)).fun

    def f(alpha):
        return _H_ng_G_alpha(nc, alpha)

    H_nc_G_alpha = minimize_scalar(f, bounds=(min_alpha, max_alpha)).fun

    def f(alpha):
        return _H_ngc_G_nc_alpha(contingency_table, alpha)

    H_ngc_G_nc_alpha = minimize_scalar(f, bounds=(min_alpha, max_alpha)).fun

    def f(alpha):
        return _H_ngc_G_nc_alpha(contingency_table.T, alpha)

    H_ncg_G_ng_alpha = minimize_scalar(f, bounds=(min_alpha, max_alpha)).fun

    def f(alpha):
        return _H_ngc_G_nc_alpha(ngg, alpha)

    H_ngg_G_ng_alpha = minimize_scalar(f, bounds=(min_alpha, max_alpha)).fun

    def f(alpha):
        return _H_ngc_G_nc_alpha(ncc, alpha)

    H_ncc_G_nc_alpha = minimize_scalar(f, bounds=(min_alpha, max_alpha)).fun
    delta_Hg = np.log(n) + H_ng_G_alpha
    delta_Hc = np.log(n) + H_nc_G_alpha
    delta_HgGc = np.log(n) + H_ngc_G_nc_alpha
    delta_HcGg = np.log(n) + H_ncg_G_ng_alpha
    delta_HgGg = np.log(n) + H_ngg_G_ng_alpha
    delta_HcGc = np.log(n) + H_ncc_G_nc_alpha
    delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = (
        delta_Hg / np.log(2),
        delta_Hc / np.log(2),
        delta_HgGc / np.log(2),
        delta_HcGg / np.log(2),
        delta_HgGg / np.log(2),
        delta_HcGc / np.log(2),
    )
    return delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc


def get_contingency_table(candidate_labels, true_labels):
    assert len(true_labels) == len(
        candidate_labels
    ), "The number of true and candidate labels must be the same."
    return pd.crosstab(
        true_labels,
        candidate_labels,
        rownames=["true_labels"],
        colnames=["candidate_labels"],
    )


def compute_RMI_from_contingency_table(
    contingency_table, reduction="DM", normalization="asymmetric", verbose=False
):
    assert reduction in [
        "DM",
        "flat",
        "none",
    ], "The reduction must be one of {'DM', 'flat', 'none'}"
    assert normalization in [
        "asymmetric",
        "symmetric",
        "none",
    ], "The normalization must be one of {'asymmetric', 'symmetric', 'none'}"
    if type(contingency_table) == pd.DataFrame:
        contingency_table = contingency_table.to_numpy()
    assert (
        type(contingency_table) == np.ndarray
    ), "The contingency table must be a numpy array or a pandas DataFrame."
    contingency_table = contingency_table[~np.all(contingency_table == 0, axis=1)]
    contingency_table = contingency_table[:, ~np.all(contingency_table == 0, axis=0)]
    I, Hg, Hc = _compute_I_Hg_Hc(contingency_table)
    HgGc = Hg - I
    HcGg = Hc - I
    HgGg = 0
    HcGc = 0
    if reduction == "none":
        delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
    if reduction == "flat":
        delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = (
            _compute_flat_subleading_terms(contingency_table)
        )
    if reduction == "DM":
        delta_Hg, delta_Hc, delta_HgGc, delta_HcGg, delta_HgGg, delta_HcGc = (
            _compute_DM_subleading_terms(contingency_table, verbose=verbose)
        )
    Hg += delta_Hg
    Hc += delta_Hc
    HgGc += delta_HgGc
    HcGg += delta_HcGg
    HgGg += delta_HgGg
    HcGc += delta_HcGc
    RMI_g_c = Hg - HgGc
    RMI_c_g = Hc - HcGg
    RMI_g_g = Hg - HgGg
    RMI_c_c = Hc - HcGc
    if normalization == "none":
        RMI = RMI_g_c
    if normalization == "asymmetric":
        RMI = RMI_g_c / RMI_g_g
    if normalization == "symmetric":
        RMI = (RMI_g_c + RMI_c_g) / (RMI_g_g + RMI_c_c)
    return RMI


def compute_RMI(
    candidate_labels,
    true_labels,
    reduction="DM",
    normalization="asymmetric",
    verbose=False,
):
    assert len(true_labels) == len(
        candidate_labels
    ), "The number of true and candidate labels must be the same."
    assert reduction in [
        "DM",
        "flat",
        "none",
    ], "The reduction must be one of {'DM', 'flat', 'none'}"
    assert normalization in [
        "asymmetric",
        "symmetric",
        "none",
    ], "The normalization must be one of {'asymmetric', 'symmetric', 'none'}"
    contingency_table = get_contingency_table(true_labels, candidate_labels)
    RMI = compute_RMI_from_contingency_table(
        contingency_table,
        reduction=reduction,
        normalization=normalization,
        verbose=verbose,
    )
    return RMI


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
        len(original_to_integer_node_id_dict), "singletonclustersalt"
    )
    current_delimiter = get_delimiter(filepath)
    with open(filepath, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            node_id, cluster_id = line.strip().split(current_delimiter)
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


def get_agri(
    groundtruth_partition,
    estimated_partition,
    edgelist,
    original_to_integer_node_id_dict,
):
    # https://codeocean.com/capsule/0712485/tree/v1
    groundtruth_edgemask = get_edge_mask(
        edgelist, groundtruth_partition, original_to_integer_node_id_dict
    )
    estimated_edgemask = get_edge_mask(
        edgelist, estimated_partition, original_to_integer_node_id_dict
    )
    gt_sum = sum(groundtruth_edgemask)
    estimated_sum = sum(estimated_edgemask)
    both_sum = sum(groundtruth_edgemask * estimated_edgemask)
    size = len(groundtruth_edgemask)
    return (both_sum - gt_sum * estimated_sum / size) / (
        0.5 * (gt_sum + estimated_sum) - gt_sum * estimated_sum / size
    )


def get_cluster_node_pairs(partition):
    n = len(partition)
    return set(
        [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if partition[i] == partition[j]
        ]
    )


def get_fnr_fpr(groundtruth_partition, estimated_partition, num_processors):
    with mp.Pool(processes=num_processors) as pool:
        results = pool.map(
            get_cluster_node_pairs, [groundtruth_partition, estimated_partition]
        )
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


def file_has_value(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
            return bool(content)
    except Exception:
        return False


def calc_precision(gt, et, matrix=None):
    if matrix is None:
        matrix = pair_confusion_matrix(gt, et)
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calc_recall(gt, et, matrix=None):
    if matrix is None:
        matrix = pair_confusion_matrix(gt, et)
    tp = matrix[1, 1]
    fn = matrix[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calc_f1_score(gt, et, matrix=None):
    if matrix is None:
        matrix = pair_confusion_matrix(gt, et)
    precision = calc_precision(gt, et, matrix)
    recall = calc_recall(gt, et, matrix)
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def calc_fnr(gt, et, matrix=None):
    if matrix is None:
        matrix = pair_confusion_matrix(gt, et)
    fn = matrix[1, 0]
    tp = matrix[1, 1]
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def calc_fpr(gt, et, matrix=None):
    if matrix is None:
        matrix = pair_confusion_matrix(gt, et)
    fp = matrix[0, 1]
    tn = matrix[0, 0]
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def clustering_accuracy(
    input_edgelist,
    groundtruth_clustering,
    estimated_clustering,
    output_prefix,
    num_processors=1,
    local=False,
    overwrite=False,
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
    if overwrite or not file_has_value(node_coverage_path):
        try:
            node_coverage = float(len(estimated_node_set)) / original_node_set_length
            with open(node_coverage_path, "w") as f:
                f.write(f"{node_coverage}\n")
        except Exception as e:
            print(f"Error writing node coverage: {e}")

    # Write NMI
    nmi_path = output_prefix + ".nmi"
    if overwrite or not file_has_value(nmi_path):
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
    if overwrite or not file_has_value(ami_path):
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
    if overwrite or not file_has_value(ari_path):
        try:
            current_ari = adjusted_rand_score(
                groundtruth_partition, estimated_partition
            )
            with open(ari_path, "w") as f:
                f.write(f"{current_ari}\n")
        except Exception as e:
            print(f"Error writing ARI: {e}")

    # Write AGRI
    agri_path = output_prefix + ".agri"
    if overwrite or not file_has_value(agri_path):
        try:
            current_agri = get_agri(
                groundtruth_partition,
                estimated_partition,
                input_edgelist,
                original_to_integer_node_id_dict,
            )
            with open(agri_path, "w") as f:
                f.write(f"{current_agri}\n")
        except Exception as e:
            print(f"Error writing AGRI: {e}")

    # Write FNR, FPR, Precision, Recall, F1-score
    # (metrics based on pair confusion matrix)
    # to avoid redundant computation, we compute the pair confusion matrix once (if needed)
    # and pass it to each function
    confusion_matrix = None

    fnr_path = output_prefix + ".fnr"
    if overwrite or not file_has_value(fnr_path):
        try:
            confusion_matrix = pair_confusion_matrix(
                groundtruth_partition, estimated_partition
            )
            current_fnr = calc_fnr(
                groundtruth_partition, estimated_partition, confusion_matrix
            )
            with open(fnr_path, "w") as f:
                f.write(f"{current_fnr}\n")
        except Exception as e:
            print(f"Error writing FNR: {e}")

    fpr_path = output_prefix + ".fpr"
    if overwrite or not file_has_value(fpr_path):
        try:
            if confusion_matrix is None:
                confusion_matrix = pair_confusion_matrix(
                    groundtruth_partition, estimated_partition
                )
            current_fpr = calc_fpr(
                groundtruth_partition, estimated_partition, confusion_matrix
            )
            with open(fpr_path, "w") as f:
                f.write(f"{current_fpr}\n")
        except Exception as e:
            print(f"Error writing FPR: {e}")

    precision_path = output_prefix + ".precision"
    if overwrite or not file_has_value(precision_path):
        try:
            if confusion_matrix is None:
                confusion_matrix = pair_confusion_matrix(
                    groundtruth_partition, estimated_partition
                )
            current_precision = calc_precision(
                groundtruth_partition, estimated_partition, confusion_matrix
            )
            with open(precision_path, "w") as f:
                f.write(f"{current_precision}\n")
        except Exception as e:
            print(f"Error writing Precision: {e}")

    recall_path = output_prefix + ".recall"
    if overwrite or not file_has_value(recall_path):
        try:
            if confusion_matrix is None:
                confusion_matrix = pair_confusion_matrix(
                    groundtruth_partition, estimated_partition
                )
            current_recall = calc_recall(
                groundtruth_partition, estimated_partition, confusion_matrix
            )
            with open(recall_path, "w") as f:
                f.write(f"{current_recall}\n")
        except Exception as e:
            print(f"Error writing Recall: {e}")

    f1_score_path = output_prefix + ".f1_score"
    if overwrite or not file_has_value(f1_score_path):
        try:
            if confusion_matrix is None:
                confusion_matrix = pair_confusion_matrix(
                    groundtruth_partition, estimated_partition
                )
            current_f1_score = calc_f1_score(
                groundtruth_partition, estimated_partition, confusion_matrix
            )
            with open(f1_score_path, "w") as f:
                f.write(f"{current_f1_score}\n")
        except Exception as e:
            print(f"Error writing F1-score: {e}")

    # # Write RMI
    # rmi_path = output_prefix + ".rmi"
    # if overwrite or not file_has_value(rmi_path):
    #     try:
    #         rmi = compute_RMI(
    #             list(estimated_partition),
    #             list(groundtruth_partition),
    #             reduction="DM",
    #             normalization="asymmetric",
    #             verbose=False,
    #         )
    #         with open(rmi_path, "w") as f:
    #             f.write(f"{rmi}\n")
    #     except Exception as e:
    #         print(f"Error writing RMI: {e}")


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
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )

    args = parser.parse_args()

    clustering_accuracy(
        args.input_network,
        args.gt_clustering,
        args.est_clustering,
        args.output_prefix,
        num_processors=args.num_processors,
        local=args.local,
        overwrite=args.overwrite,
    )
