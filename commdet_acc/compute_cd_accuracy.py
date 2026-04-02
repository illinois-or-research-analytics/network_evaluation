import math
import os
import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import graph_tool.all as gt
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix


# ==========================================
# State Tracking
# ==========================================
class StateTracker:
    """Tracks computation state by maintaining a sha256sum compatible 'done' file via atomic writes."""

    def __init__(self, output_dir, input_files):
        self.output_dir = Path(output_dir)
        self.done_file = self.output_dir / "done"
        self.input_files = [str(f) for f in input_files if f and os.path.exists(f)]
        self.current_input_hashes = {
            f: self._compute_file_hash(f) for f in self.input_files
        }
        self.completed_outputs = self._load_state()

    def _compute_file_hash(self, filepath):
        if not os.path.exists(filepath):
            return None
        hasher = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return None

    def _load_state(self):
        completed = {}
        if not self.done_file.exists():
            return completed

        recorded_hashes = {}
        try:
            with open(self.done_file, "r") as f:
                for line in f:
                    parts = line.strip().split("  ", 1)
                    if len(parts) == 2:
                        recorded_hashes[parts[1]] = parts[0]
        except Exception:
            pass

        inputs_match = True
        for f, h in self.current_input_hashes.items():
            if recorded_hashes.get(f) != h:
                inputs_match = False
                break

        if not inputs_match:
            return completed

        for filepath, h in recorded_hashes.items():
            if filepath not in self.current_input_hashes:
                completed[filepath] = h

        return completed

    def is_done(self, output_filepath):
        out_path = str(output_filepath)
        if out_path not in self.completed_outputs:
            return False
        current_hash = self._compute_file_hash(out_path)
        if current_hash is None:
            return False
        if self.completed_outputs[out_path] != current_hash:
            return False
        return True

    def mark_done(self, output_filepath):
        out_path = str(output_filepath)
        h = self._compute_file_hash(out_path)
        if h is None:
            return  # File disappeared before hashing

        self.completed_outputs[out_path] = h
        self._write_state_atomically()

    def _write_state_atomically(self):
        tmp_file = self.done_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                for filepath, h in self.current_input_hashes.items():
                    f.write(f"{h}  {filepath}\n")
                for filepath, h in self.completed_outputs.items():
                    f.write(f"{h}  {filepath}\n")
            tmp_file.replace(self.done_file)
        except Exception as e:
            logging.warning(f"Failed to atomically write state tracking file: {e}")
            if tmp_file.exists():
                tmp_file.unlink()


# ==========================================
# Constants
# ==========================================
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

EXPECTED_METRICS = [
    "node_coverage",
    "nmi",
    "ami",
    "ari",
    "fnr",
    "fpr",
    "precision",
    "recall",
    "f1_score",
]


# ==========================================
# I/O and Logging Helpers
# ==========================================
def prepare_logging(output_dir):
    logging.basicConfig(
        filename=Path(output_dir) / "run.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(logging.StreamHandler(sys.stdout))


def get_delimiter(filepath):
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if "," in line:
                return ","
            if "\t" in line:
                return "\t"
            if " " in line:
                return " "
            break
    return ","


def load_edgelist_df(filepath):
    delimiter = get_delimiter(filepath)
    df = pd.read_csv(
        filepath, sep=delimiter, header=None, comment="#", dtype=str, na_filter=False
    )
    if not df.empty and len(df.columns) >= 2:
        val0, val1 = (
            str(df.iloc[0, 0]).strip().lower(),
            str(df.iloc[0, 1]).strip().lower(),
        )
        if val0 in NODE_COLUMN_NAMES and val1 in NODE_COLUMN_NAMES:
            df = df.iloc[1:].reset_index(drop=True)
    return df


def load_clustering_df(filepath):
    delimiter = get_delimiter(filepath)
    df = pd.read_csv(
        filepath, sep=delimiter, header=None, comment="#", dtype=str, na_filter=False
    )
    if not df.empty and len(df.columns) >= 2:
        val0, val1 = (
            str(df.iloc[0, 0]).strip().lower(),
            str(df.iloc[0, 1]).strip().lower(),
        )
        if val0 in NODE_COLUMN_NAMES and val1 in COMMUNITY_COLUMN_NAMES:
            df = df.iloc[1:].reset_index(drop=True)
    return df


def get_node_set_edgelist(edgelist):
    df = load_edgelist_df(edgelist)
    if df.empty:
        return set()
    return set(df[0]).union(set(df[1]))


def get_node_set_clustering(filepath):
    df = load_clustering_df(filepath)
    if df.empty:
        return set()
    counts = df.groupby(1)[0].count()
    valid_clusters = counts[counts > 1].index
    return set(df[df[1].isin(valid_clusters)][0])


def create_mapping(node_set):
    return {node_id: idx for idx, node_id in enumerate(node_set)}


def read_clustering(filepath, original_to_integer_node_id_dict):
    current_partition = np.full(
        len(original_to_integer_node_id_dict), "singletonclustersalt", dtype=object
    )
    df = load_clustering_df(filepath)

    for row in df.itertuples(index=False):
        node_id, cluster_id = str(row[0]), str(row[1])
        if node_id in original_to_integer_node_id_dict:
            current_partition[original_to_integer_node_id_dict[node_id]] = cluster_id

    current_integer_cluster_id = 0
    raw_cluster_to_integer_id_dict = dict()
    for i, raw_cluster_id in enumerate(current_partition):
        if raw_cluster_id == "singletonclustersalt":
            current_partition[i] = current_integer_cluster_id
            current_integer_cluster_id += 1
        else:
            if raw_cluster_id not in raw_cluster_to_integer_id_dict:
                raw_cluster_to_integer_id_dict[raw_cluster_id] = (
                    current_integer_cluster_id
                )
                current_integer_cluster_id += 1
            current_partition[i] = raw_cluster_to_integer_id_dict[raw_cluster_id]
    return current_partition.astype(int)


def cleanup_extra_files(output_dir, expected_filenames):
    out_path = Path(output_dir)
    if not out_path.exists():
        return
    for file_path in out_path.iterdir():
        if file_path.is_file():
            fname = file_path.name
            if fname in expected_filenames:
                continue
            if fname.endswith(".log") or fname == "done":
                continue
            try:
                file_path.unlink()
            except Exception as e:
                logging.warning(f"Failed to remove extra file {fname}: {e}")


# ==========================================
# Metric Computations
# ==========================================
def get_confusion_matrix(gt_partition, est_partition, matrix=None):
    """Computes the pair confusion matrix."""
    # Note that TP, TN, FP, FN are twice the actual counts due to symmetry.
    # Please see scikit-learn's pair_confusion_matrix documentation for details.
    if matrix is None:
        return pair_confusion_matrix(gt_partition, est_partition)
    return matrix


def calc_precision(matrix):
    return (
        matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
        if (matrix[1, 1] + matrix[0, 1]) > 0
        else 0.0
    )


def calc_recall(matrix):
    return (
        matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
        if (matrix[1, 1] + matrix[1, 0]) > 0
        else 0.0
    )


def calc_f1_score(matrix):
    p, r = calc_precision(matrix), calc_recall(matrix)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def calc_fnr(matrix):
    return (
        matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
        if (matrix[1, 0] + matrix[1, 1]) > 0
        else 0.0
    )


def calc_fpr(matrix):
    return (
        matrix[0, 1] / (matrix[0, 1] + matrix[0, 0])
        if (matrix[0, 1] + matrix[0, 0]) > 0
        else 0.0
    )


# ==========================================
# Main Execution
# ==========================================
def clustering_accuracy(
    input_edgelist,
    groundtruth_clustering,
    estimated_clustering,
    output_prefix,
    num_processors=1,
    local=False,
):
    out_dir = os.path.dirname(output_prefix) or "."
    prefix_base = os.path.basename(output_prefix)
    os.makedirs(out_dir, exist_ok=True)
    prepare_logging(out_dir)

    tracker = StateTracker(
        out_dir, [input_edgelist, groundtruth_clustering, estimated_clustering]
    )
    expected_files = {f"{prefix_base}.{m}" for m in EXPECTED_METRICS}

    if all(tracker.is_done(f"{output_prefix}.{m}") for m in EXPECTED_METRICS):
        logging.info("All specified accuracy metrics already computed. Exiting early.")
        cleanup_extra_files(out_dir, expected_files)
        return

    logging.info("Starting accuracy computation...")
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

    if "node_coverage" in EXPECTED_METRICS and not tracker.is_done(
        f"{output_prefix}.node_coverage"
    ):
        try:
            logging.info("Computing node_coverage...")
            val = float(len(estimated_node_set)) / original_node_set_length
            if math.isnan(val):
                logging.warning("node_coverage is NaN")
            pd.Series([val]).to_csv(
                f"{output_prefix}.node_coverage", index=False, header=False, na_rep="NaN"
            )
            tracker.mark_done(f"{output_prefix}.node_coverage")
        except Exception as e:
            logging.error(f"Error computing node_coverage: {e}")

    if "nmi" in EXPECTED_METRICS and not tracker.is_done(f"{output_prefix}.nmi"):
        try:
            logging.info("Computing nmi...")
            val = gt.mutual_information(
                groundtruth_partition, estimated_partition, norm=True, adjusted=False
            )
            if math.isnan(val):
                logging.warning("nmi is NaN")
            pd.Series([val]).to_csv(f"{output_prefix}.nmi", index=False, header=False, na_rep="NaN")
            tracker.mark_done(f"{output_prefix}.nmi")
        except Exception as e:
            logging.error(f"Error computing nmi: {e}")

    if "ami" in EXPECTED_METRICS and not tracker.is_done(f"{output_prefix}.ami"):
        try:
            logging.info("Computing ami...")
            val = gt.mutual_information(
                groundtruth_partition, estimated_partition, adjusted=True
            )
            if math.isnan(val):
                logging.warning("ami is NaN")
            pd.Series([val]).to_csv(f"{output_prefix}.ami", index=False, header=False, na_rep="NaN")
            tracker.mark_done(f"{output_prefix}.ami")
        except Exception as e:
            logging.error(f"Error computing ami: {e}")

    if "ari" in EXPECTED_METRICS and not tracker.is_done(f"{output_prefix}.ari"):
        try:
            logging.info("Computing ari...")
            val = adjusted_rand_score(groundtruth_partition, estimated_partition)
            if math.isnan(val):
                logging.warning("ari is NaN")
            pd.Series([val]).to_csv(f"{output_prefix}.ari", index=False, header=False, na_rep="NaN")
            tracker.mark_done(f"{output_prefix}.ari")
        except Exception as e:
            logging.error(f"Error computing ari: {e}")

    cm_funcs = {
        "fnr": calc_fnr,
        "fpr": calc_fpr,
        "precision": calc_precision,
        "recall": calc_recall,
        "f1_score": calc_f1_score,
    }
    cm_needed = [
        m
        for m in cm_funcs
        if m in EXPECTED_METRICS and not tracker.is_done(f"{output_prefix}.{m}")
    ]

    if cm_needed:
        try:
            logging.info(
                f"Computing Confusion Matrix for metrics: {', '.join(cm_needed)}..."
            )
            confusion_matrix = get_confusion_matrix(
                groundtruth_partition, estimated_partition
            )
            for name in cm_needed:
                try:
                    val = cm_funcs[name](confusion_matrix)
                    if math.isnan(val):
                        logging.warning(f"{name} is NaN")
                    pd.Series([val]).to_csv(
                        f"{output_prefix}.{name}", index=False, header=False, na_rep="NaN"
                    )
                    tracker.mark_done(f"{output_prefix}.{name}")
                except Exception as e:
                    logging.error(f"Error computing {name}: {e}")
        except Exception as e:
            logging.error(f"Error generating confusion matrix: {e}")

    cleanup_extra_files(out_dir, expected_files)
    logging.info("Accuracy computation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute clustering accuracy metrics.")
    parser.add_argument(
        "--input-network", help="Path to the input edgelist file", required=True
    )
    parser.add_argument(
        "--gt-clustering", help="Path to the groundtruth clustering file", required=True
    )
    parser.add_argument(
        "--est-clustering", help="Path to the estimated clustering file", required=True
    )
    parser.add_argument(
        "--output-prefix",
        help="Path to the output prefix (no extension)",
        required=True,
    )
    parser.add_argument("--num_processors", type=int, default=1)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Constrain evaluation to nodes present in est-clustering",
    )
    args = parser.parse_args()

    clustering_accuracy(
        args.input_network,
        args.gt_clustering,
        args.est_clustering,
        args.output_prefix,
        args.num_processors,
        args.local,
    )
