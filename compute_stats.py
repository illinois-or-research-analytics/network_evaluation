import csv
import json
from pathlib import Path

import click
import networkit as nk
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List

from hm01.graph import Graph, IntangibleSubgraph
from hm01.mincut import viecut

import time
import logging
from scipy.sparse import dok_matrix
import psutil
import os

STATS_JSON_FILENAME = 'stats.json'
NODE_ORDERING_IDX_FILENAME = 'node_ordering.idx'
CLUSTER_ORDERING_IDX_FILENAME = 'cluster_ordering.idx'
OUTLIER_ORDERING_IDX_FILENAME = 'outlier_ordering.idx'
OUTLIERS_TSV_FILENAME = 'outliers.tsv'
STATS_LOG_FILENAME = 'stats_log.log'


@click.command()
@click.option(
    '--input-network',
    required=True,
    type=click.Path(exists=True),
    help='Input network',
)
@click.option(
    '--input-clustering',
    required=True,
    type=click.Path(exists=True),
    help='Input clustering',
)
@click.option(
    '--output-folder',
    required=True,
    type=click.Path(),
    help='Ouput folder',
)
@click.option(
    '--overwrite',
    is_flag=True,
    help='Whether to overwrite existing data',
)
def compute_stats(input_network, input_clustering, output_folder, overwrite):
    # Prepare output folder
    dir_path = Path(output_folder)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Start logging
    logging.basicConfig(
        filename=os.path.join(output_folder, STATS_LOG_FILENAME),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    job_start_time = time.time()

    # Read the network
    log_cpu_ram_usage("Start")
    logging.info("Reading input network!")
    start_time = time.time()
    elr = nk.graphio.EdgeListReader(
        '\t',
        0,
        continuous=False,
        directed=False,
    )
    graph = elr.read(input_network)
    graph.removeMultiEdges()
    graph.removeSelfLoops()
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After reading network!")

    # id is the real id (str), iid is the internal id (int)

    # Generate the node mapping
    logging.info("Generating mappings and orderings.")
    start_time = time.time()
    node_mapping_dict = elr.getNodeMap()
    node_mapping_dict_reversed = {
        v: k
        for k, v in node_mapping_dict.items()
    }
    # node_mapping_dict: {node_id: node_iid}
    # node_mapping_dict_reversed: {node_iid: node_id}
    node_order = list(graph.iterNodes())  # list of node_iids

    # Generate the node ordering
    with open(dir_path / NODE_ORDERING_IDX_FILENAME, 'w') as idx_f:
        node_ordering_idx_list = [
            [node_mapping_dict_reversed[node_iid]]
            for node_iid in node_order
        ]
        df = pd.DataFrame(node_ordering_idx_list)
        df.to_csv(idx_f, sep='\t', header=False, index=False)

    # Generate the cluster mapping
    clustering_dict, cluster_mapping_dict = read_clustering(input_clustering)
    cluster_mapping_dict_reversed = {
        v: k
        for k, v in cluster_mapping_dict.items()
    }
    # clustering_dict: {node_id: cluster_iid}
    # cluster_mapping_dict: {cluster_id: cluster_iid}
    # cluster_mapping_dict_reversed: {cluster_iid: cluster_id}
    cluster_order = list(cluster_mapping_dict.values())  # list of cluster_iids

    # Generate the cluster ordering
    with open(dir_path / CLUSTER_ORDERING_IDX_FILENAME, 'w') as idx_f:
        cluster_ordering_idx_list = [
            [cluster_mapping_dict_reversed[cluster_iid]]
            for cluster_iid in cluster_order
        ]
        df = pd.DataFrame(cluster_ordering_idx_list)
        df.to_csv(idx_f, sep='\t', header=False, index=False)

    # Generate the outlier ordering
    outlier_nodes, clustered_nodes = \
        get_outliers(graph, node_mapping_dict, clustering_dict)
    outlier_order = list(outlier_nodes)  # [node_iid]

    with open(dir_path / OUTLIER_ORDERING_IDX_FILENAME, 'w') as idx_f:
        outlier_ordering_idx_list = [
            [node_mapping_dict_reversed[node_iid]]
            for node_iid in outlier_order
        ]
        df = pd.DataFrame(outlier_ordering_idx_list)
        df.to_csv(idx_f, sep='\t', header=False, index=False)

    with open(dir_path / OUTLIERS_TSV_FILENAME, 'w') as idx_f:
        outlier_tsv_list = [
            [node]
            for node in outlier_nodes
        ]
        df = pd.DataFrame(outlier_tsv_list)
        df.to_csv(idx_f, sep='\t', header=False, index=False)

    o_subgraph = nk.graphtools.subgraphFromNodes(graph, outlier_nodes)
    c_subgraph = nk.graphtools.subgraphFromNodes(graph, clustered_nodes)

    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After generating orderings and mapping!")

    logging.info("Stats - Number of nodes and edges!")
    start_time = time.time()
    # S1 - number of nodes
    n_nodes = compute_n_nodes(graph)

    # S2 - number of edges
    n_edges = compute_n_edges(graph)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After S1 and S2!")

    logging.info("Stats - S3 and S4!")
    start_time = time.time()
    # S3 and S4 - number of connected components and connected components size distribution
    n_concomp, concomp_sizes_distr = get_cc_stats(graph)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After connected components!")

    logging.info("Stats - S5!")
    start_time = time.time()
    # S5 - degree assortativity
    deg_assort = compute_deg_assort(graph)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After degree assortativity!")

    logging.info("Stats - S6!")
    start_time = time.time()
    # S6 - Global Clustering Coefficient
    global_ccoeff = compute_global_ccoeff(graph)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After global clustering coefficient!")

    logging.info("Stats - S8 and S9!")
    start_time = time.time()
    # S8 and S9 - degree distribution, degree sequence
    deg_distr = compute_deg_distr(graph, node_order)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After degree distribution!")

    # TODO: S12?

    logging.info("Stats - S21")
    start_time = time.time()
    # S21 - diameter
    diameter = compute_diameter(graph)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After diameter!")

    # S23 - Jaccard similarity
    # TODO: this has not been implemented

    logging.info("Stats - S13 - S16!")
    start_time = time.time()
    # S13 number of outliers
    n_onodes = len(outlier_nodes)

    # S14 number of edges among outliers nodes
    o_o_edges = o_subgraph.numberOfEdges()

    # S15 number of edges between outlier and non-outlier nodes
    o_no_edges = n_edges - o_o_edges - c_subgraph.numberOfEdges()

    # S16 degree distribution for the outlier node subgraph
    osub_deg_distr = [
        o_subgraph.degree(u)
        for u in outlier_order
    ]

    # TODO: any S?
    # outlier degree distribution
    o_deg_distr = [
        graph.degree(u)
        for u in outlier_order
    ]

    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After outlier stats!")

    # S17 degree distribution for edges that connect outlier nodes to non-outlier nodes
    # TODO: Should this distribution include outlier-outlier edges?
    # TODO: this has not been implemented

    # Getting cluster statistics

    logging.info("Stats - Cluster stats!")
    start_time = time.time()

    cluster_stats = \
        compute_cluster_stats(
            input_network,
            input_clustering,
            cluster_mapping_dict_reversed,
            cluster_order,
        )
    cluster_stats.to_csv(dir_path / 'cluster_stats.csv', index=False)

    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After minimum cut!")

    n_clusters = len(cluster_stats)

    # S10 and S11 - Cluster size distribution

    c_n_nodes_distr = cluster_stats['n'].values
    c_n_edges_distr = cluster_stats['m'].values

    logging.info("Stats - S18!")
    start_time = time.time()

    # S18 - number of disconnected clusters
    n_disconnected_clusters = \
        int((cluster_stats['connectivity'] < 1).sum())
    ratio_disconnected_clusters = \
        n_disconnected_clusters / n_clusters

    n_wellconnected_clusters = \
        int((cluster_stats['connectivity_normalized_log10(n)'] > 1).sum())
    ratio_wellconnected_clusters = \
        n_wellconnected_clusters / n_clusters

    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After connectivity!")

    # S19 and S20 - mininum cut size distribution, mincut sequence
    mincuts_distr = cluster_stats['connectivity'].values

    logging.info("Stats - S22!")
    start_time = time.time()

    # S22 - mixing parameter
    mixing_mu_distr, mixing_xi = compute_mixing_params(
        graph,
        clustering_dict,
        node_mapping_dict_reversed,
        node_order,
    )

    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After mixing parameters!")

    # S24 - Participation coefficient distribution
    # TODO: test this
    logging.info("Stats - S24!")
    start_time = time.time()

    participation_coeffs, o_participation_coeffs_distr = \
        compute_participation_coeff_distr(
            graph,
            node_mapping_dict_reversed,
            clustering_dict,
            node_order,
            outlier_order,
        )

    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After participation coefficients!")

    # Save scalar statistics
    logging.info("Saving Scalar Stats!")
    start_time = time.time()

    stats_to_save = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'n_concomp': n_concomp,
        'deg_assort': deg_assort,
        'global_ccoeff': global_ccoeff,
        'diameter': diameter,

        'n_onodes': n_onodes,
        'o_o_edges': o_o_edges,
        'o_no_edges': o_no_edges,

        'n_clusters': n_clusters,
        'n_disconnects': n_disconnected_clusters,
        'ratio_disconnected_clusters': ratio_disconnected_clusters,
        'n_wellconnected_clusters': n_wellconnected_clusters,
        'ratio_wellconnected_clusters': ratio_wellconnected_clusters,
        'mixing_xi': mixing_xi,
    }
    save_scalar_stats(dir_path, stats_to_save, overwrite)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After Saving scalar stats!")

    # Save distribution statistics

    logging.info("Saving Distribution Stats!")
    start_time = time.time()

    distr_stats_dict = {
        'degree': deg_distr,
        'concomp_sizes': concomp_sizes_distr,

        'osub_degree': osub_deg_distr,
        'o_deg': o_deg_distr,

        'c_size': c_n_nodes_distr,
        'c_edges': c_n_edges_distr,
        'mincuts': mincuts_distr,
        'mixing_mus': mixing_mu_distr,
        'participation_coeffs': participation_coeffs,
        'o_participation_coeffs': o_participation_coeffs_distr,
    }
    save_distr_stats(overwrite, dir_path, distr_stats_dict)
    logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
    log_cpu_ram_usage("After Distribution scalar stats!")

    logging.info(f"Total Time taken: {round(
        time.time() - job_start_time, 3)} seconds")
    log_cpu_ram_usage("Usage statistics after job completion!")


def log_cpu_ram_usage(step_name):
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    logging.info(f"Step: {step_name} | CPU Usage: {cpu_percent}% | RAM Usage: {
                 ram_percent}% | Disk Usage: {disk_percent}")


def compute_cluster_stats(input_network, input_clustering, cluster_mapping_dict_reversed, cluster_order):
    clusters = load_clusters(input_clustering)  # {cluster_id: subgraph}
    edgelist_reader = nk.graphio.EdgeListReader('\t', 0)
    nk_graph = edgelist_reader.read(input_network)
    global_graph = Graph(nk_graph, '')

    n_nodes_cluster = [
        clusters[cluster_mapping_dict_reversed[cluster_id]].n()
        for cluster_id in cluster_order
    ]

    n_edges_cluster = [
        clusters[
            cluster_mapping_dict_reversed[cluster_id]
        ].count_edges(global_graph)
        for cluster_id in cluster_order
    ]

    clusters = {
        cluster_id: clusters[
            cluster_mapping_dict_reversed[cluster_id]
        ].realize(global_graph)
        for cluster_id in cluster_order
    }

    mincuts = [
        viecut(clusters[cluster_id])[-1]
        for cluster_id in cluster_order
    ]
    mincuts_normalized = [
        mincut / np.log10(n)
        for mincut, n in zip(mincuts, n_nodes_cluster)
    ]

    cluster_stats = pd.DataFrame(
        list(
            zip(
                cluster_order,
                n_nodes_cluster,
                n_edges_cluster,
                mincuts,
                mincuts_normalized,
            )
        ),
        columns=[
            'cluster',
            'n',
            'm',
            'connectivity',
            'connectivity_normalized_log10(n)',
        ]
    )

    return cluster_stats


def save_distr_stats(overwrite, dir_path, distr_stats_dict):
    distribution_arr = (dir_path / 'distributions').glob('*.distribution')
    distribution_name_arr = [
        Path(current_distribution_file).stem
        for current_distribution_file in distribution_arr
    ]

    for distr_stat in distr_stats_dict.keys():
        if f'{distr_stat}.distribution' in distribution_name_arr and not overwrite:
            continue

        with open(dir_path / f'{distr_stat}.distribution', 'w') as distr_f:
            distr_stat_list = [
                [v]
                for v in distr_stats_dict.get(distr_stat)
            ]
            df = pd.DataFrame(distr_stat_list)
            df.to_csv(distr_f, sep='\t', header=False, index=False)


def save_scalar_stats(dir_path, stats_to_save, overwrite):
    stats_file = dir_path / STATS_JSON_FILENAME
    file_rw_bit = 'w'

    stats_dict = {}
    if stats_file.is_file():
        file_rw_bit = 'r'
        with stats_file.open(file_rw_bit) as f:
            stats_dict = json.load(f)

    for stat, value in stats_to_save.items():
        if stat not in stats_dict or overwrite:
            stats_dict[stat] = value

    with stats_file.open('w') as f:
        json.dump(stats_dict, f, indent=4)


def compute_n_edges(graph):
    return graph.numberOfEdges()


def compute_n_nodes(graph):
    return graph.numberOfNodes()


def compute_global_ccoeff(graph):
    return nk.globals.ClusteringCoefficient.exactGlobal(graph)


def compute_deg_assort(graph):
    # convert from NetworKit.Graph to networkx.Graph
    nx_graph = nk.nxadapter.nk2nx(graph)
    deg_assort = nx.degree_assortativity_coefficient(nx_graph)
    return deg_assort


def read_clustering(filepath):
    cluster_df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=[
            'node_id',
            'cluster_name',
        ],
        dtype=str,
    )
    unique_values = cluster_df['cluster_name'].unique()
    value_map = {
        value: idx
        for idx, value in enumerate(unique_values)
    }
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)
    clustering_dict = dict(
        zip(
            cluster_df['node_id'],
            cluster_df['cluster_id'],
        )
    )
    return clustering_dict, value_map


def get_outliers(graph, node_mapping, clustering_dict):
    # node_mapping: {node_id: node_iid}
    # clustering_dict: {node_id: cluster_iid}
    clustered_nodes = [
        node_mapping[u]
        for u in clustering_dict.keys()
    ]
    nodes_set = set(graph.iterNodes())
    outlier_nodes = nodes_set.difference(clustered_nodes)
    # clustered_nodes: [node_iid]
    # outlier_nodes: {node_iid}
    return outlier_nodes, clustered_nodes


def compute_deg_distr(graph, node_order):
    return [
        graph.degree(v)
        for v in node_order
    ]


def get_cc_stats(graph):
    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    num_cc = cc.numberOfComponents()
    cc_size_distribution = cc.getComponentSizes()
    return num_cc, cc_size_distribution.values()


def compute_cluster_size_distr(clustering_dict):
    cluster_size_dict = {}
    for cluster in clustering_dict.values():
        cluster_size_dict[cluster] = cluster_size_dict.get(cluster, 0) + 1
    cluster_size_distr = []
    for i in range(len(cluster_size_dict.keys())):
        cluster_size_distr.append(cluster_size_dict.get(i))
    return cluster_size_distr


def compute_mixing_params(graph, clustering_dict, node_mapping_dict_reversed, node_order):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for node1, node2 in graph.iterEdges():
        n1 = str(node_mapping_dict_reversed.get(node1))
        n2 = str(node_mapping_dict_reversed.get(node2))
        if n1 not in clustering_dict or n2 not in clustering_dict:
            continue
        if clustering_dict[n1] == clustering_dict[n2]:  # nodes are co-clustered
            in_degree[node1] += 1
            in_degree[node2] += 1
        else:
            out_degree[node1] += 1
            out_degree[node2] += 1
    mus = [
        out_degree[i]/(out_degree[i] + in_degree[i])
        if (out_degree[i] + in_degree[i]) != 0
        else 0
        for i in node_order
    ]

    outs = [out_degree[i] for i in graph.iterNodes()]
    xi = np.sum(outs) / 2 / (graph.numberOfEdges())

    return mus, xi


def compute_diameter(graph):
    start = time.time()
    connected_graph = \
        nk.components.ConnectedComponents.extractLargestConnectedComponent(
            graph, True)
    print(f"Time taken to extract largest connected component: {
          time.time() - start}")
    print(connected_graph.numberOfNodes())
    print(connected_graph.numberOfEdges())

    start = time.time()
    diam = nk.distance.Diameter(connected_graph, algo=1)
    diam.run()
    diameter = diam.getDiameter()
    print(f"Time taken to compute diameter: {time.time() - start}")

    return diameter[0]


def get_participation_coeffs(graph, clustering_dict, node_mapping_dict_reversed):
    participation_dict = defaultdict(dict)
    for v in graph.iterNodes():
        for neighbor in graph.iterNeighbors(v):
            neighbor_cluster = \
                clustering_dict.get(node_mapping_dict_reversed[neighbor], -1)
            participation_dict[v].setdefault(neighbor_cluster, 0)
            participation_dict[v][neighbor_cluster] += 1

        if graph.isIsolated(v):
            participation_dict[v] = {-1: 0}

    return participation_dict


def compute_participation_coeff_distr(graph, node_mapping_dict_reversed, clustering_dict, node_order, outlier_order):
    participation_dict = \
        get_participation_coeffs(
            graph,
            clustering_dict,
            node_mapping_dict_reversed,
        )

    participation_coeffs = []
    outlier_participation_coeffs = {}
    for node in node_order:
        participation = participation_dict[node]
        deg_of_node = sum(participation.values())

        coeff = 1.0
        if deg_of_node > 0:
            coeff -= \
                np.sum([
                    (deg_i / deg_of_node) ** 2
                    for deg_i in list(participation.values())
                ])
            if -1 in participation.keys():
                coeff += (participation[-1] / deg_of_node) ** 2
                coeff -= participation[-1] * ((1 / deg_of_node) ** 2)

        participation_coeffs.append(coeff)
        if node in outlier_order:
            outlier_participation_coeffs[node] = coeff

    o_participation_coeffs_distr = [
        outlier_participation_coeffs.get(v)
        for v in outlier_order
    ]

    return participation_coeffs, o_participation_coeffs_distr


def load_clusters(filepath, cluster_iid2id, cluster_order) -> List[IntangibleSubgraph]:
    clusters: Dict[str, IntangibleSubgraph] = {}
    with open(filepath) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for line in csv_reader:
            node_id, cluster_id = line
            clusters.setdefault(
                cluster_id, IntangibleSubgraph([], cluster_id)
            ).subset.append(int(node_id))
    return [
        clusters[cluster_iid2id[cluster_iid]]
        for cluster_iid in cluster_order
    ]


def compute_cluster_stats(network_fp, clustering_fp, cluster_iid2id, cluster_order):
    clusters = \
        load_clusters(
            clustering_fp,
            cluster_iid2id,
            cluster_order,
        )
    ids = [
        cluster.index
        for cluster in clusters
    ]
    ns = [
        cluster.n()
        for cluster in clusters
    ]

    # TODO: check this reader
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
    nk_graph = edgelist_reader.read(network_fp)

    global_graph = Graph(nk_graph, "")
    ms = [
        cluster.count_edges(global_graph)
        for cluster in clusters
    ]

    # modularities = [
    #     global_graph.modularity_of(cluster)
    #     for cluster in clusters
    # ]

    clusters = [
        cluster.realize(global_graph)
        for cluster in clusters
    ]

    mincuts = [
        viecut(cluster)[-1]
        for cluster in clusters
    ]
    mincuts_normalized = [
        mincut / np.log10(ns[i])
        for i, mincut in enumerate(mincuts)
    ]
    # mincuts_normalized_log2 = [
    #     mincut / np.log2(ns[i])
    #     for i, mincut in enumerate(mincuts)
    # ]
    # mincuts_normalized_sqrt = [
    #     mincut / (ns[i] ** 0.5 / 5)
    #     for i, mincut in enumerate(mincuts)
    # ]

    # conductances = [
    #     cluster.conductance(global_graph)
    #     for cluster in clusters
    # ]

    # m = global_graph.m()
    # ids.append("Overall")
    # modularities.append(sum(modularities))

    # ns.append(global_graph.n())
    # ms.append(m)
    # mincuts.append(None)
    # mincuts_normalized.append(None)
    # mincuts_normalized_log2.append(None)
    # mincuts_normalized_sqrt.append(None)
    # conductances.append(None)

    df = pd.DataFrame(
        list(
            zip(
                ids,
                ns,
                ms,
                # modularities,
                mincuts,
                mincuts_normalized,
                # mincuts_normalized_log2,
                # mincuts_normalized_sqrt,
                # conductances
            )
        ),
        columns=[
            'cluster',
            'n',
            'm',
            # 'modularity',
            'connectivity',
            'connectivity_normalized_log10(n)',
            # 'connectivity_normalized_log2(n)',
            # 'connectivity_normalized_sqrt(n)/5',
            # 'conductance',
        ]
    )

    return df


if __name__ == '__main__':
    compute_stats()
