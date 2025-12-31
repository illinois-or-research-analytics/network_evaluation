import json
import argparse
from pathlib import Path
import logging
import time

import numpy as np
import graph_tool.all as gt
from scipy.sparse import linalg as la


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
    print("[WARNING] Could not detect delimiter...")
    return ","


def compute_statistic(stat_name, compute_fn, gt_stats, overwrite=False):
    if overwrite or stat_name not in gt_stats:
        logging.info(f"Compute {stat_name}")
        start_time = time.perf_counter()
        result = compute_fn()
        elapsed = time.perf_counter() - start_time
        gt_stats[stat_name] = result
        logging.info(f"{stat_name}: {elapsed}")
    else:
        result = gt_stats[stat_name]
    return result


def n_edges(G):
    return G.num_edges()


def mean_degree(G):
    return np.mean([v.out_degree() for v in G.vertices()])


def degree_assortativity(G):
    deg_assort, _ = gt.scalar_assortativity(G, "total")
    return deg_assort


def mean_kcore_value(G):
    core = gt.kcore_decomposition(G)
    return np.mean([core[v] for v in G.vertices()])


def local_clustering_coefficient(G):
    local_ccoeffs = gt.local_clustering(G)
    return np.mean([local_ccoeffs[v] for v in G.vertices()])


def global_clustering_coefficient(G):
    global_ccoeff, _ = gt.global_clustering(G)
    return global_ccoeff


def leading_eigenvalue_adjacency(G):
    largest_eigval_A, _ = gt.eigenvector(G)
    return largest_eigval_A


def leading_eigenvalue_hashimoto(G):
    H_mtx = gt.hashimoto(G)
    eigvals_H = la.eigs(H_mtx, k=1, return_eigenvectors=False, which="LR")
    return eigvals_H[0].real


def characteristic_time_random_walk(G):
    largest_cc = gt.extract_largest_component(G)
    T = gt.transition(largest_cc)
    eigvals_T = la.eigs(T, k=2, return_eigenvectors=False, which="LR")
    sorted_eigvals = np.sort(eigvals_T.real)
    second_eigval = sorted_eigvals[-2]
    assert second_eigval > 0, f"Non-positive second eigenvalue: {second_eigval}"
    return -1 / np.log(second_eigval)


def pseudo_diameter(G):
    pseudo_diameter, *_ = gt.pseudo_diameter(G)
    return pseudo_diameter


def node_percolation_targeted(G):
    vertices = sorted(
        [v for v in G.vertices()], key=lambda v: v.out_degree(), reverse=True
    )
    sizes, _ = gt.vertex_percolation(G, vertices)
    return np.mean(sizes) / G.num_vertices()


def node_percolation_random(G):
    n_trials = 5
    Rr = 0.0
    vertices = [v for v in G.vertices()]
    for i in range(n_trials):
        np.random.shuffle(vertices)
        sizes2, _ = gt.vertex_percolation(G, vertices)
        Rr += np.mean(sizes2) / G.num_vertices() / n_trials
    return Rr


def fraction_giant_component(G):
    giant_cc = gt.extract_largest_component(G)
    return giant_cc.num_vertices() / G.num_vertices()


parser = argparse.ArgumentParser(description="Compute excess edges in a graph.")
parser.add_argument(
    "--network",
    type=str,
    help="Path to the network file",
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Path to the output directory",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing statistics",
)

args = parser.parse_args()
network_fp = Path(args.network)
output_dir = Path(args.outdir)
is_overwrite = args.overwrite

assert network_fp.exists(), f"File not found: {network_fp}"
assert network_fp.is_file(), f"Not a file: {network_fp}"

output_dir.mkdir(parents=True, exist_ok=True)
log_path = output_dir / "gt_stats.log"
logging.basicConfig(
    filename=log_path,
    filemode="w" if is_overwrite else "a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info(f"Input: {network_fp}")
logging.info(f"Output: {output_dir}")

# If gt_stats.json exists, load it
gt_stats_fp = output_dir / "gt_stats.json"
if gt_stats_fp.exists():
    with open(gt_stats_fp, "r") as f:
        gt_stats = json.load(f)
else:
    gt_stats = dict()

statistics_functions = {
    "n_edges": n_edges,
    "mean_degree": mean_degree,
    "deg_assort": degree_assortativity,
    "mean_kcore": mean_kcore_value,
    "local_ccoeff": local_clustering_coefficient,
    "global_ccoeff": global_clustering_coefficient,
    "l_eigval_A": leading_eigenvalue_adjacency,
    "l_eigval_H": leading_eigenvalue_hashimoto,
    "char_time": characteristic_time_random_walk,
    "pseudo_diameter": pseudo_diameter,
    "node_percolation_targeted": node_percolation_targeted,
    "node_percolation_random": node_percolation_random,
    "frac_giant_ccomp": fraction_giant_component,
}

# Check if all statistics are already computed
if not is_overwrite:
    missing_stats = [stat for stat in statistics_functions if stat not in gt_stats]
    if not missing_stats:
        logging.info("All statistics are already computed.")
        exit(0)

# Load the graph
start = time.perf_counter()
G = gt.load_graph_from_csv(
    str(network_fp),
    directed=False,
    csv_options={"delimiter": detect_delimiter(network_fp)},
)
gt.remove_parallel_edges(G)
gt.remove_self_loops(G)
elapsed = time.perf_counter() - start
logging.info(f"Loaded graph: {elapsed}")

# Compute the statistics
start = time.perf_counter()
for stat_name, compute_fn in statistics_functions.items():
    gt_stats[stat_name] = compute_statistic(
        stat_name,
        lambda fn=compute_fn: fn(G),
        gt_stats,
        overwrite=is_overwrite,
    )

    # Save the computed statistics
    with open(gt_stats_fp, "w") as f:
        json.dump(gt_stats, f, indent=4)
elapsed = time.perf_counter() - start
logging.info(f"Computed all statistics: {elapsed}")
