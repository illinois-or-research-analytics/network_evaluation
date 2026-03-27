import argparse
from pathlib import Path
import hashlib
import logging
import os
import sys

import pandas as pd
from pymincut.pygraph import PyGraph


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
            return

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

EXPECTED_STATS = [
    "global_n",
    "global_m",
    "node_coverage",
    "n_outliers",
    "mixing_parameter",
    "n",
    "m",
    "c",
    "conductance",
    "degree_density",
    "edge_density",
    "mincut",
    "modularity",
    "n_clusters",
    "n_disconnected_clusters",
    "n_connected_clusters",
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


def detect_delimiter(file_path):
    with open(file_path, "r") as f:
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
    logging.warning(f"Could not detect delimiter for {file_path}, defaulting to ','")
    return ","


def load_edgelist_df(filepath):
    delimiter = detect_delimiter(filepath)
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
    delimiter = detect_delimiter(filepath)
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


def load_cached_list(filepath):
    with open(filepath, "r") as f:
        return [
            (
                float(line.strip())
                if "." in line.strip() or "e" in line.strip().lower()
                else int(float(line.strip()))
            )
            for line in f
            if line.strip()
        ]


# ==========================================
# Data Processing & Mapping Helpers
# ==========================================
def build_community_mappings(df_com):
    node_id2iid, node_iid2id, node_iid_count = {}, {}, 0
    com_id2iid, com_iid2id, com_iid_count = {}, {}, 0
    node2coms_raw, com2nodes_raw = {}, {}

    for row in df_com.itertuples(index=False):
        node_id, com_id = str(row[0]), str(row[1])

        if node_id not in node_id2iid:
            node_id2iid[node_id] = node_iid_count
            node_iid2id[node_iid_count] = node_id
            node_iid_count += 1
        if com_id not in com_id2iid:
            com_id2iid[com_id] = com_iid_count
            com_iid2id[com_iid_count] = com_id
            com_iid_count += 1

        node2coms_raw.setdefault(node_id2iid[node_id], set()).add(com_id2iid[com_id])
        com2nodes_raw.setdefault(com_id2iid[com_id], set()).add(node_id2iid[node_id])

    valid_old_com_iids = [
        cid for cid in sorted(com2nodes_raw.keys()) if len(com2nodes_raw[cid]) > 1
    ]
    new_com_id2iid, new_com_iid2id, new_com2nodes, new_node2coms = {}, {}, {}, {}
    new_com_iid_count = 0

    for old_iid in valid_old_com_iids:
        com_id_str = com_iid2id[old_iid]
        new_com_id2iid[com_id_str] = new_com_iid_count
        new_com_iid2id[new_com_iid_count] = com_id_str
        nodes = com2nodes_raw[old_iid]
        new_com2nodes[new_com_iid_count] = nodes
        for node_iid in nodes:
            new_node2coms.setdefault(node_iid, set()).add(new_com_iid_count)
        new_com_iid_count += 1

    return (
        node_id2iid,
        node_iid2id,
        node_iid_count,
        new_com_id2iid,
        new_com_iid2id,
        new_com2nodes,
        new_node2coms,
        new_com_iid_count,
    )


def build_adjacency(df_net, node_id2iid, node_iid2id, node_iid_count):
    neighbors = {}
    outliers = set()

    for row in df_net.itertuples(index=False):
        node1, node2 = str(row[0]), str(row[1])
        if node1 == node2:
            continue

        node1_iid = node_id2iid.get(node1)
        node2_iid = node_id2iid.get(node2)

        if (
            node1_iid is not None
            and node2_iid is not None
            and node2_iid in neighbors.get(node1_iid, set())
        ):
            continue

        if node1 not in node_id2iid:
            outliers.add(node_iid_count)
            node_id2iid[node1] = node_iid_count
            node_iid2id[node_iid_count] = node1
            node_iid_count += 1
        if node2 not in node_id2iid:
            outliers.add(node_iid_count)
            node_id2iid[node2] = node_iid_count
            node_iid2id[node_iid_count] = node2
            node_iid_count += 1

        neighbors.setdefault(node_id2iid[node1], set()).add(node_id2iid[node2])
        neighbors.setdefault(node_id2iid[node2], set()).add(node_id2iid[node1])

    return neighbors, outliers, node_id2iid, node_iid2id, node_iid_count


# ==========================================
# Metric Computations
# ==========================================
def compute_global_n(neighbors):
    return len(neighbors)


def compute_global_m(neighbors):
    return sum(len(neighbors[node]) for node in neighbors) // 2


def compute_n_outliers(outliers):
    return len(outliers)


def compute_node_coverage(outliers, global_n=None):
    global_n = compute_global_n(neighbors) if global_n is None else global_n
    n_outliers = len(outliers)
    return 1 - n_outliers / global_n if n_outliers > 0 else 1.0


def compute_mS_cS(neighbors, com):
    m_count = 0
    c_count = 0
    for node in com:
        for neighbor in neighbors.get(node, []):
            if neighbor in com:
                m_count += 1
            else:
                c_count += 1
    return m_count // 2, c_count


def compute_nS(com):
    return len(com)


def compute_conductance(neighbors, com, m=None, c=None):
    m, c = compute_mS_cS(neighbors, com) if m is None or c is None else (m, c)
    return c / (2 * m + c) if c > 0 else 0.0


def compute_degree_density(neighbors, com, m=None, n=None):
    m = compute_mS_cS(neighbors, com)[0] if m is None else m
    n = compute_nS(com) if n is None else n
    return m / n if m > 0 else 0.0


def compute_edge_density(neighbors, com, m=None, n=None):
    m = compute_mS_cS(neighbors, com)[0] if m is None else m
    n = compute_nS(com) if n is None else n
    return 2 * m / (n * (n - 1)) if m > 0 else 0.0


def compute_mincut(neighbors, com):
    cluster_edges = set()
    for node in com:
        for neighbor in neighbors.get(node, []):
            if neighbor in com:
                cluster_edges.add((node, neighbor))
    sub_G = PyGraph(list(com), list(cluster_edges))
    return sub_G.mincut("noi", "bqueue", False)[2]


def compute_modularity(neighbors, com, global_m=None, m=None, c=None):
    m, c = compute_mS_cS(neighbors, com) if m is None or c is None else (m, c)
    global_m = compute_global_m(neighbors) if global_m is None else global_m
    return (m / global_m) - ((2 * m + c) / (2 * global_m)) ** 2


def compute_mixing_parameter(node, neighbors, node2coms, outliers):
    if node in outliers:
        return 1.0
    assert node in node2coms, f"Node {node} not found in node2coms mapping."
    coms = node2coms[node]
    n_neighbors = 0
    n_in = 0
    for neighbor in neighbors.get(node, set()):
        n_neighbors += 1
        n_in += 1 if coms.intersection(node2coms.get(neighbor, set())) else 0
    return 1.0 - n_in / n_neighbors if n_in > 0 else 0.0


def compute_n_clusters(n_clusters):
    return n_clusters


def compute_n_disconnected_clusters(cluster_stats):
    return sum(
        1
        for n, mc in zip(cluster_stats["n"], cluster_stats["mincut"])
        if n > 1 and mc == 0
    )


def compute_n_connected_clusters(cluster_stats):
    return sum(
        1
        for n, mc in zip(cluster_stats["n"], cluster_stats["mincut"])
        if n > 1 and mc > 0
    )


# ==========================================
# Main Execution
# ==========================================
def compute_cluster_stats(network_file, community_file, outdir):
    network_fp = Path(network_file)
    community_fp = Path(community_file)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_logging(output_dir)

    tracker = StateTracker(output_dir, [network_fp, community_fp])
    expected_files = {f"{m}.txt" for m in EXPECTED_STATS}.union(
        {"node.idx", "com.idx", "outliers.idx"}
    )

    all_done = all(
        tracker.is_done(str(output_dir / f"{k}.txt")) for k in EXPECTED_STATS
    )
    if (
        all_done
        and tracker.is_done(str(output_dir / "node.idx"))
        and tracker.is_done(str(output_dir / "com.idx"))
        and tracker.is_done(str(output_dir / "outliers.idx"))
    ):
        logging.info("All expected cluster stats already computed. Exiting early.")
        cleanup_extra_files(output_dir, expected_files)
        return

    logging.info("Loading community data...")
    df_com = load_clustering_df(community_fp)
    (
        node_id2iid,
        node_iid2id,
        node_iid_count,
        com_id2iid,
        com_iid2id,
        com2nodes,
        node2coms,
        com_iid_count,
    ) = build_community_mappings(df_com)

    outliers = {
        node_iid for node_iid in range(node_iid_count) if node_iid not in node2coms
    }

    logging.info("Loading network data...")
    df_net = load_edgelist_df(network_fp)
    neighbors, net_outliers, node_id2iid, node_iid2id, node_iid_count = build_adjacency(
        df_net, node_id2iid, node_iid2id, node_iid_count
    )
    outliers.update(net_outliers)

    idx_node = str(output_dir / "node.idx")
    if not tracker.is_done(idx_node):
        pd.Series([node_iid2id[iid] for iid in range(node_iid_count)]).to_csv(
            idx_node, index=False, header=False
        )
        tracker.mark_done(idx_node)

    idx_com = str(output_dir / "com.idx")
    if not tracker.is_done(idx_com):
        pd.Series([com_iid2id[iid] for iid in range(com_iid_count)]).to_csv(
            idx_com, index=False, header=False
        )
        tracker.mark_done(idx_com)

    idx_outliers = str(output_dir / "outliers.idx")
    if not tracker.is_done(idx_outliers):
        pd.Series([node_iid2id[iid] for iid in sorted(outliers)]).to_csv(
            idx_outliers, index=False, header=False
        )
        tracker.mark_done(idx_outliers)

    logging.info("Computing global stats...")
    global_stats = {}
    global_n = compute_global_n(neighbors)
    global_m = compute_global_m(neighbors)

    global_stats["global_n"] = (
        global_n
        if not tracker.is_done(str(output_dir / "global_n.txt"))
        else load_cached_list(output_dir / "global_n.txt")[0]
    )
    global_stats["global_m"] = (
        global_m
        if not tracker.is_done(str(output_dir / "global_m.txt"))
        else load_cached_list(output_dir / "global_m.txt")[0]
    )
    global_stats["node_coverage"] = (
        compute_node_coverage(outliers, global_n)
        if not tracker.is_done(str(output_dir / "node_coverage.txt"))
        else load_cached_list(output_dir / "node_coverage.txt")[0]
    )
    global_stats["n_outliers"] = (
        compute_n_outliers(outliers)
        if not tracker.is_done(str(output_dir / "n_outliers.txt"))
        else load_cached_list(output_dir / "n_outliers.txt")[0]
    )

    logging.info("Computing node-level mixing parameters...")
    node_stats = {"mixing_parameter": []}
    if not tracker.is_done(str(output_dir / "mixing_parameter.txt")):
        for node_iid in range(node_iid_count):
            node_stats["mixing_parameter"].append(
                compute_mixing_parameter(node_iid, neighbors, node2coms, outliers)
            )
    else:
        node_stats["mixing_parameter"] = load_cached_list(
            output_dir / "mixing_parameter.txt"
        )

    logging.info("Computing cluster-level structural properties...")
    cluster_stats = {
        "m": [],
        "n": [],
        "c": [],
        "conductance": [],
        "degree_density": [],
        "edge_density": [],
        "mincut": [],
        "modularity": [],
    }
    compute_flags = {
        k: not tracker.is_done(str(output_dir / f"{k}.txt"))
        for k in cluster_stats.keys()
    }

    for k, needs_compute in compute_flags.items():
        if not needs_compute:
            cluster_stats[k] = load_cached_list(output_dir / f"{k}.txt")

    for com_iid in range(com_iid_count):
        assert (
            com_iid in com2nodes
        ), f"Community IID {com_iid} not found in com2nodes mapping."
        com = com2nodes[com_iid]

        n = (
            compute_nS(com)
            if any(
                [
                    compute_flags["n"],
                    compute_flags["degree_density"],
                    compute_flags["edge_density"],
                ]
            )
            else None
        )
        m_c = (
            compute_mS_cS(neighbors, com)
            if any(
                [
                    compute_flags["m"],
                    compute_flags["c"],
                    compute_flags["conductance"],
                    compute_flags["degree_density"],
                    compute_flags["edge_density"],
                    compute_flags["modularity"],
                ]
            )
            else (None, None)
        )
        m, c = m_c

        if compute_flags["n"]:
            cluster_stats["n"].append(n)
        if compute_flags["m"]:
            cluster_stats["m"].append(m)
        if compute_flags["c"]:
            cluster_stats["c"].append(c)
        if compute_flags["conductance"]:
            cluster_stats["conductance"].append(
                compute_conductance(neighbors, com, m, c)
            )
        if compute_flags["degree_density"]:
            cluster_stats["degree_density"].append(
                compute_degree_density(neighbors, com, m, n)
            )
        if compute_flags["edge_density"]:
            cluster_stats["edge_density"].append(
                compute_edge_density(neighbors, com, m, n)
            )
        if compute_flags["mincut"]:
            cluster_stats["mincut"].append(compute_mincut(neighbors, com))
        if compute_flags["modularity"]:
            cluster_stats["modularity"].append(
                compute_modularity(neighbors, com, global_m, m, c)
            )

    global_stats["n_clusters"] = (
        compute_n_clusters(com_iid_count)
        if not tracker.is_done(str(output_dir / "n_clusters.txt"))
        else load_cached_list(output_dir / "n_clusters.txt")[0]
    )
    global_stats["n_disconnected_clusters"] = (
        compute_n_disconnected_clusters(cluster_stats)
        if not tracker.is_done(str(output_dir / "n_disconnected_clusters.txt"))
        else load_cached_list(output_dir / "n_disconnected_clusters.txt")[0]
    )
    global_stats["n_connected_clusters"] = (
        compute_n_connected_clusters(cluster_stats)
        if not tracker.is_done(str(output_dir / "n_connected_clusters.txt"))
        else load_cached_list(output_dir / "n_connected_clusters.txt")[0]
    )

    all_stats = {**global_stats, **node_stats, **cluster_stats}

    for key in EXPECTED_STATS:
        stat_file = str(output_dir / f"{key}.txt")
        if key in all_stats and not tracker.is_done(stat_file):
            value = all_stats[key]
            pd.Series(value if isinstance(value, list) else [value]).to_csv(
                stat_file, index=False, header=False
            )
            tracker.mark_done(stat_file)

    cleanup_extra_files(output_dir, expected_files)
    logging.info("Cluster stats computation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cluster-level network stats.")
    parser.add_argument("--network", type=str, required=True, help="Edgelist CSV")
    parser.add_argument(
        "--community", type=str, required=True, help="Community assignment CSV"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    compute_cluster_stats(args.network, args.community, args.outdir)
