import argparse
from pathlib import Path
import logging
import time
import sys
import hashlib
import os

import numpy as np
import pandas as pd
import graph_tool.all as gt
from scipy.sparse import linalg as la


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
# Constants & Configuration
# ==========================================
NODE_ORDERING_IDX_FILENAME = "node.idx"

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

EXPECTED_SCALARS = [
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

EXPECTED_DISTR = ["concomp_sizes", "degree", "local_ccoeff_nodes", "pagerank", "kcore"]


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
    logging.warning("Could not detect delimiter, defaulting to ','")
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
# Caching Helpers (Graph-Tool)
# ==========================================
def get_out_degrees(G, cache):
    if "out_degrees" not in cache:
        cache["out_degrees"] = G.get_out_degrees(G.get_vertices())
    return cache["out_degrees"]


def get_components(G, cache):
    if "components" not in cache:
        _, cache["components"] = gt.label_components(G)
    return cache["components"]


def get_local_clustering(G, cache):
    if "local_clustering" not in cache:
        cache["local_clustering"] = gt.local_clustering(G).a
    return cache["local_clustering"]


def get_largest_cc(G, cache):
    if "largest_cc" not in cache:
        cache["largest_cc"] = gt.extract_largest_component(G, prune=False)
    return cache["largest_cc"]


def get_kcore(G, cache):
    if "kcore" not in cache:
        cache["kcore"] = gt.kcore_decomposition(G).a
    return cache["kcore"]


# ==========================================
# Metric Computations
# ==========================================
def compute_n_nodes(G, cache):
    return int(G.num_vertices())


def compute_n_edges(G, cache):
    return int(G.num_edges())


def compute_mean_degree(G, cache):
    return float(np.mean(get_out_degrees(G, cache)))


def compute_n_concomp(G, cache):
    return int(len(get_components(G, cache)))


def compute_local_ccoeff_mean(G, cache):
    return float(np.mean(get_local_clustering(G, cache)))


def compute_deg_assort(G, cache):
    return float(gt.scalar_assortativity(G, "total")[0])


def compute_mean_kcore(G, cache):
    return float(np.mean(get_kcore(G, cache)))


def compute_global_ccoeff(G, cache):
    return float(gt.global_clustering(G)[0])


def compute_pseudo_diameter(G, cache):
    return float(gt.pseudo_diameter(G)[0])


def compute_char_time(G, cache):
    T = gt.transition(gt.extract_largest_component(G, prune=True))
    eigvals_T = la.eigs(T, k=2, return_eigenvectors=False, which="LR")
    return float(-1 / np.log(np.sort(eigvals_T.real)[-2]))


def compute_percolation_targeted(G, cache):
    vertices = sorted(
        [v for v in G.vertices()], key=lambda v: v.out_degree(), reverse=True
    )
    sizes, _ = gt.vertex_percolation(G, vertices)
    return float(np.mean(sizes) / G.num_vertices())


def compute_percolation_random(G, cache):
    n_trials = 5
    Rr = 0.0
    vertices = list(G.vertices())
    for _ in range(n_trials):
        np.random.shuffle(vertices)
        sizes2, _ = gt.vertex_percolation(G, vertices)
        Rr += np.mean(sizes2) / G.num_vertices() / n_trials
    return float(Rr)


def compute_frac_giant_comp(G, cache):
    return float(get_largest_cc(G, cache).num_vertices() / G.num_vertices())


def compute_degree_dist(G, cache):
    return get_out_degrees(G, cache)


def compute_concomp_sizes(G, cache):
    return get_components(G, cache).tolist()


def compute_local_ccoeff_dist(G, cache):
    return get_local_clustering(G, cache)


def compute_pagerank_dist(G, cache):
    return gt.pagerank(G).a


def compute_kcore_dist(G, cache):
    return get_kcore(G, cache)


SCALAR_DISPATCH = {
    "n_nodes": compute_n_nodes,
    "n_edges": compute_n_edges,
    "n_concomp": compute_n_concomp,
    "mean_degree": compute_mean_degree,
    "deg_assort": compute_deg_assort,
    "mean_kcore": compute_mean_kcore,
    "global_ccoeff": compute_global_ccoeff,
    "local_ccoeff": compute_local_ccoeff_mean,
    "pseudo_diameter": compute_pseudo_diameter,
    "char_time": compute_char_time,
    "node_percolation_targeted": compute_percolation_targeted,
    "node_percolation_random": compute_percolation_random,
    "frac_giant_ccomp": compute_frac_giant_comp,
}

DISTR_DISPATCH = {
    "degree": compute_degree_dist,
    "concomp_sizes": compute_concomp_sizes,
    "local_ccoeff_nodes": compute_local_ccoeff_dist,
    "pagerank": compute_pagerank_dist,
    "kcore": compute_kcore_dist,
}


# ==========================================
# Main Execution
# ==========================================
def compute_stats(input_network, output_dir):
    job_start_time = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_logging(output_dir)

    tracker = StateTracker(output_dir, [input_network])

    expected_files = {f"{m}.txt" for m in EXPECTED_SCALARS}.union(
        {f"{m}.txt" for m in EXPECTED_DISTR}
    )
    expected_files.add(NODE_ORDERING_IDX_FILENAME)

    all_done = True
    for stat_name in EXPECTED_SCALARS:
        if not tracker.is_done(str(output_dir / f"{stat_name}.txt")):
            all_done = False
            break
    for stat_name in EXPECTED_DISTR:
        if not tracker.is_done(str(output_dir / f"{stat_name}.txt")):
            all_done = False
            break

    if all_done and tracker.is_done(str(output_dir / NODE_ORDERING_IDX_FILENAME)):
        logging.info("All expected stats already computed. Exiting early.")
        cleanup_extra_files(output_dir, expected_files)
        return

    computation_cache = {}
    logging.info("Building graph via hashed edge list...")
    start_time = time.perf_counter()

    df = load_edgelist_df(input_network)
    G = gt.Graph(directed=False)
    v_name = G.add_edge_list(df.iloc[:, [0, 1]].values.astype(str), hashed=True)
    G.vertex_properties["name"] = v_name
    gt.remove_parallel_edges(G)
    gt.remove_self_loops(G)
    logging.info(f"Graph built in {time.perf_counter() - start_time:.3f}s")

    idx_file = str(output_dir / NODE_ORDERING_IDX_FILENAME)
    if not tracker.is_done(idx_file):
        start_time = time.perf_counter()
        pd.Series([v_name[v] for v in G.vertices()]).to_csv(
            idx_file, index=False, header=False
        )
        tracker.mark_done(idx_file)
        logging.info(f"node.idx saved in {time.perf_counter() - start_time:.3f}s")

    for stat_name in EXPECTED_SCALARS:
        stat_file = str(output_dir / f"{stat_name}.txt")
        if tracker.is_done(stat_file):
            continue
        logging.info(f"Computing {stat_name}...")
        start = time.perf_counter()
        val = SCALAR_DISPATCH[stat_name](G, computation_cache)
        pd.Series([val]).to_csv(stat_file, index=False, header=False, na_rep="NaN")
        tracker.mark_done(stat_file)
        logging.info(f"{stat_name} completed in {time.perf_counter() - start:.3f}s")

    for stat_name in EXPECTED_DISTR:
        distr_file = str(output_dir / f"{stat_name}.txt")
        if tracker.is_done(distr_file):
            continue
        logging.info(f"Computing {stat_name}...")
        start = time.perf_counter()
        data = DISTR_DISPATCH[stat_name](G, computation_cache)
        pd.DataFrame(data).to_csv(
            distr_file, sep="\t", header=False, index=False, na_rep="NaN"
        )
        tracker.mark_done(distr_file)
        logging.info(f"{stat_name} completed in {time.perf_counter() - start:.3f}s")

    logging.info(f"Total time taken: {time.perf_counter() - job_start_time:.3f}s")
    cleanup_extra_files(output_dir, expected_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", required=True, type=str, help="Path to edge list CSV"
    )
    parser.add_argument(
        "--outdir", required=True, type=str, help="Path to output directory"
    )
    args = parser.parse_args()
    compute_stats(args.network, args.outdir)
