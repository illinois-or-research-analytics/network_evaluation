import json
from pathlib import Path

import networkit as nk
import pandas as pd

import time
import logging
import sys
import argparse

STATS_JSON_FILENAME = "stats.json"
NODE_ORDERING_IDX_FILENAME = "node_ordering.idx"

SCALAR_STATS = {
    "n_nodes",
    "n_edges",
    "n_concomp",
    "deg_assort",
    "global_ccoeff",
    "local_ccoeff",
    "diameter",
}

DISTR_STATS = {
    "degree",
    "concomp_sizes",
}


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


def compute_stats(input_network, output_dir, overwrite):
    # TODO: globally caching some intermediate results
    # TODO: refactor by abstracting the statistics
    # TODO: better profile memory and CPU usage

    job_start_time = time.perf_counter()

    # Prepare output folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start logging
    prepare_logging(output_dir, overwrite)

    # Prepare agenda
    stats_to_compute = SCALAR_STATS | DISTR_STATS

    if not overwrite:
        existing_scalar_stats_file = output_dir / STATS_JSON_FILENAME
        existing_scalar_stats_dict = {}
        if existing_scalar_stats_file.is_file():
            with existing_scalar_stats_file.open("r") as f:
                existing_scalar_stats_dict = json.load(f)
        stats_to_compute -= set(existing_scalar_stats_dict.keys())

        existing_distr_stats_files = output_dir.glob("*.distribution")
        existing_distr_stats_names = [
            Path(existing_distr_stats_file).stem
            for existing_distr_stats_file in existing_distr_stats_files
        ]
        stats_to_compute -= set(existing_distr_stats_names)

    scalar_stats = {}
    distr_stats = {}

    # Determine the delimiter
    delimiter = detect_delimiter(input_network)

    # Read the network
    logging.info("Reading input network")
    start_time = time.perf_counter()

    elr = nk.graphio.EdgeListReader(delimiter, 0, continuous=False, directed=False)
    graph = elr.read(input_network)
    graph.removeMultiEdges()
    graph.removeSelfLoops()

    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    logging.info("Generating node mapping and ordering.")
    start_time = time.perf_counter()

    # Generate the node mapping
    node_mapping_dict = elr.getNodeMap()
    node_mapping_dict_reversed = {v: k for k, v in node_mapping_dict.items()}
    node_order = list(graph.iterNodes())

    # Generate the node ordering
    with open(output_dir / NODE_ORDERING_IDX_FILENAME, "w") as idx_f:
        node_ordering_idx_list = [
            [node_mapping_dict_reversed[node_iid]] for node_iid in node_order
        ]
        df = pd.DataFrame(node_ordering_idx_list)
        df.to_csv(idx_f, sep="\t", header=False, index=False)

    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Number of nodes
    logging.info("Stats - Number of nodes")
    start_time = time.perf_counter()
    if "n_nodes" in stats_to_compute:
        n_nodes = compute_n_nodes(graph)
        scalar_stats["n_nodes"] = n_nodes
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Number of edges
    logging.info("Stats - Number of edges")
    start_time = time.perf_counter()
    if "n_edges" in stats_to_compute:
        n_edges = compute_n_edges(graph)
        scalar_stats["n_edges"] = n_edges
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Number of connected components and connected components size distribution
    logging.info(
        "Stats - Number of connected components and connected components size distribution"
    )
    start_time = time.perf_counter()
    if "n_concomp" in stats_to_compute or "concomp_sizes" in stats_to_compute:
        n_concomp, concomp_sizes_distr = get_cc_stats(graph)
        scalar_stats["n_concomp"] = n_concomp
        distr_stats["concomp_sizes"] = concomp_sizes_distr
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Degree assortativity
    logging.info("Stats - Degree assortativity")
    start_time = time.perf_counter()
    if "deg_assort" in stats_to_compute:
        deg_assort = compute_deg_assort(graph)
        scalar_stats["deg_assort"] = deg_assort
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # S6 - Global Clustering Coefficient
    logging.info("Stats - Global clustering coefficient")
    start_time = time.perf_counter()
    if "global_ccoeff" in stats_to_compute:
        global_ccoeff = compute_global_ccoeff(graph)
        scalar_stats["global_ccoeff"] = global_ccoeff
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Average Local Clustering Coefficient
    logging.info("Stats - Average Local clustering coefficient")
    start_time = time.perf_counter()
    if "local_ccoeff" in stats_to_compute:
        local_ccoeff = compute_local_ccoeff(graph)
        scalar_stats["local_ccoeff"] = local_ccoeff
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Degree distribution
    logging.info("Stats - Degree distribution")
    start_time = time.perf_counter()
    if "degree" in stats_to_compute:
        deg_distr = compute_deg_distr(graph, node_order)
        distr_stats["degree"] = deg_distr
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Diameter
    logging.info("Stats - Diameter")
    start_time = time.perf_counter()
    if "diameter" in stats_to_compute:
        diameter = compute_diameter(graph)
        scalar_stats["diameter"] = diameter
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Save scalar statistics
    logging.info("Saving scalar statistics")
    start_time = time.perf_counter()
    save_scalar_stats(output_dir, scalar_stats, overwrite)
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    # Save distribution statistics
    logging.info("Saving distribution statistics")
    start_time = time.perf_counter()
    save_distr_stats(output_dir, distr_stats, overwrite)
    logging.info(f"Time taken: {time.perf_counter() - start_time:.3f} seconds")

    logging.info(
        f"Total time taken: {time.perf_counter() - job_start_time:.3f} seconds"
    )

    # Save done file
    (output_dir / "done").touch()


def prepare_logging(output_dir, is_overwrite):
    logging.basicConfig(
        filename=output_dir / "run.log",
        filemode="w" if is_overwrite else "a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def save_distr_stats(output_dir, distr_stats_dict, overwrite):
    distribution_arr = output_dir.glob("*.distribution")
    distribution_name_arr = [
        Path(current_distribution_file).stem
        for current_distribution_file in distribution_arr
    ]

    for distr_stat in distr_stats_dict.keys():
        if f"{distr_stat}.distribution" in distribution_name_arr and not overwrite:
            continue

        with open(output_dir / f"{distr_stat}.distribution", "w") as distr_f:
            distr_stat_list = [[v] for v in distr_stats_dict.get(distr_stat)]
            df = pd.DataFrame(distr_stat_list)
            df.to_csv(distr_f, sep="\t", header=False, index=False)


def save_scalar_stats(output_dir, stats_to_save, overwrite):
    stats_file = output_dir / STATS_JSON_FILENAME
    stats_dict = {}
    if stats_file.is_file():
        with stats_file.open("r") as f:
            stats_dict = json.load(f)

    for stat, value in stats_to_save.items():
        if stat not in stats_dict or overwrite:
            stats_dict[stat] = value

    with stats_file.open("w") as f:
        json.dump(stats_dict, f, indent=4)


def compute_n_edges(graph):
    return graph.numberOfEdges()


def compute_n_nodes(graph):
    return graph.numberOfNodes()


def compute_global_ccoeff(graph):
    return nk.globals.ClusteringCoefficient.exactGlobal(graph)


def compute_local_ccoeff(graph):
    return nk.globals.ClusteringCoefficient.sequentialAvgLocal(graph)


def compute_deg_assort(graph):
    """Compute degree assortativity using NetworKit."""
    # Compute degree for every node
    dc = nk.centrality.DegreeCentrality(graph)
    dc.run()
    # Compute correlation
    assortativity = nk.correlation.Assortativity(graph, dc.scores())
    assortativity.run()
    return assortativity.getCoefficient()


def compute_deg_distr(graph, node_order):
    return [graph.degree(v) for v in node_order]


def get_cc_stats(graph):
    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    num_cc = cc.numberOfComponents()
    cc_size_distribution = cc.getComponentSizes()
    return num_cc, cc_size_distribution.values()


def compute_diameter(graph):
    if graph.numberOfNodes() == 0:
        return 0
    connected_graph = (
        nk.components.ConnectedComponents.extractLargestConnectedComponent(graph, True)
    )
    diam = nk.distance.Diameter(connected_graph, algo=1)
    diam.run()
    diameter = diam.getDiameter()
    return diameter[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute statistics for a network and clustering."
    )
    parser.add_argument(
        "--network",
        required=True,
        type=str,
        help="Input network",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Output folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing data",
    )

    args = parser.parse_args()

    compute_stats(
        args.network,
        args.outdir,
        args.overwrite,
    )
