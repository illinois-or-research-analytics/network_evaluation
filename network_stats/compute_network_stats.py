import json
import argparse
from pathlib import Path
import logging
import time
import sys

import numpy as np
import pandas as pd
import graph_tool.all as gt
from scipy.sparse import linalg as la

STATS_JSON_FILENAME = "stats.json"
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

SCALAR_STATS = {
    "n_nodes",
    "n_edges",
    "n_concomp",
    "mean_degree",
    "deg_assort",
    "mean_kcore",
    "global_ccoeff",
    "local_ccoeff",
    "pseudo_diameter",
    # "l_eigval_A",
    # "l_eigval_H",
    "char_time",
    "node_percolation_targeted",
    "node_percolation_random",
    "frac_giant_ccomp",
}

DISTR_STATS = {
    "concomp_sizes",
    "degree",
    "local_ccoeff_nodes",
    "pagerank",
    "kcore",
    # "betweenness",
}

# --- IO Helpers ---


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


def check_if_header_exists(filepath, delimiter):
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.strip().split(delimiter)
            if len(parts) >= 2 and parts[0].lower() in NODE_COLUMN_NAMES:
                return True
            return False
    return False


def prepare_logging(output_dir):
    logging.basicConfig(
        filename=output_dir / "run.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# --- Caching Helpers ---


def get_out_degrees(G, cache):
    if "out_degrees" not in cache:
        cache["out_degrees"] = G.get_out_degrees(G.get_vertices())
    return cache["out_degrees"]


def get_components(G, cache):
    if "components" not in cache:
        _, hist = gt.label_components(G)
        cache["components"] = hist
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


# --- Metric Functions ---


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


def compute_l_eigval_A(G, cache):
    return float(gt.eigenvector(G)[0])


def compute_l_eigval_H(G, cache):
    H_mtx = gt.hashimoto(G)
    eigvals_H = la.eigs(H_mtx, k=1, return_eigenvectors=False, which="LR")
    return float(eigvals_H[0].real)


def compute_char_time(G, cache):
    largest_cc_view = gt.extract_largest_component(G, prune=True)
    T = gt.transition(largest_cc_view)
    eigvals_T = la.eigs(T, k=2, return_eigenvectors=False, which="LR")
    sorted_eigvals = np.sort(eigvals_T.real)
    return float(-1 / np.log(sorted_eigvals[-2]))


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


def compute_betweenness_dist(G, cache):
    return gt.betweenness(G)[0].a


def compute_kcore_dist(G, cache):
    return get_kcore(G, cache)


# --- Dispatchers ---

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
    "l_eigval_A": compute_l_eigval_A,
    "l_eigval_H": compute_l_eigval_H,
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
    "betweenness": compute_betweenness_dist,
    "kcore": compute_kcore_dist,
}

# --- Main Pipeline ---


def compute_stats(input_network, output_dir):
    job_start_time = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_logging(output_dir)

    stats_to_compute = SCALAR_STATS | DISTR_STATS
    scalar_stats = {}
    distr_stats = {}
    computation_cache = {}

    delimiter = detect_delimiter(input_network)
    has_header = check_if_header_exists(input_network, delimiter)
    header_arg = 0 if has_header else None

    logging.info("Building graph via hashed edge list...")
    start_time = time.perf_counter()

    df = pd.read_csv(
        input_network, sep=delimiter, header=header_arg, comment="#", dtype=str
    )
    edge_list = df.iloc[:, [0, 1]].values.astype(str)

    G = gt.Graph(directed=False)
    v_name = G.add_edge_list(edge_list, hashed=True)
    G.vertex_properties["name"] = v_name

    gt.remove_parallel_edges(G)
    gt.remove_self_loops(G)
    logging.info(f"Graph built in {time.perf_counter() - start_time:.3f}s")

    logging.info("Saving canonical node ordering (node.idx)...")
    start_time = time.perf_counter()
    node_names_ordered = [v_name[v] for v in G.vertices()]
    with open(output_dir / NODE_ORDERING_IDX_FILENAME, "w") as idx_f:
        pd.Series(node_names_ordered).to_csv(idx_f, index=False, header=False)
    logging.info(f"node.idx saved in {time.perf_counter() - start_time:.3f}s")

    for stat_name, compute_fn in SCALAR_DISPATCH.items():
        if stat_name in stats_to_compute:
            logging.info(f"Computing {stat_name}...")
            start = time.perf_counter()
            scalar_stats[stat_name] = compute_fn(G, computation_cache)
            logging.info(f"{stat_name} completed in {time.perf_counter() - start:.3f}s")

    for stat_name, compute_fn in DISTR_DISPATCH.items():
        if stat_name in stats_to_compute:
            logging.info(f"Computing {stat_name}...")
            start = time.perf_counter()
            distr_stats[stat_name] = compute_fn(G, computation_cache)
            logging.info(f"{stat_name} completed in {time.perf_counter() - start:.3f}s")

    if scalar_stats:
        stats_file = output_dir / STATS_JSON_FILENAME
        with stats_file.open("w") as f:
            json.dump(scalar_stats, f, indent=4)

    if distr_stats:
        for stat_name, data in distr_stats.items():
            distr_file = output_dir / f"{stat_name}.distribution"
            with open(distr_file, "w") as f:
                pd.DataFrame(data).to_csv(f, sep="\t", header=False, index=False)

    logging.info(f"Total time taken: {time.perf_counter() - job_start_time:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for a network.")
    parser.add_argument("--network", required=True, type=str, help="Input network file")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    compute_stats(args.network, args.outdir)
