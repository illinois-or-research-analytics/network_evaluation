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
import psutil
import os

STATS_JSON_FILENAME = 'stats.json'
NODE_ORDERING_IDX_FILENAME = 'node_ordering.idx'
CLUSTER_ORDERING_IDX_FILENAME = 'cluster_ordering.idx'
OUTLIER_ORDERING_IDX_FILENAME = 'outlier_ordering.idx'
OUTLIERS_TSV_FILENAME = 'outliers.tsv'
STATS_LOG_FILENAME = 'stats_log.log'

SCALAR_STATS = {
    'n_nodes',
    'n_edges',
    'n_concomp',
    'deg_assort',
    'global_ccoeff',
    'local_ccoeff',
    'diameter',

    'n_onodes',
    'o_o_edges',
    'o_no_edges',

    'n_clusters',
    'n_disconnects',
    'ratio_disconnected_clusters',
    'n_wellconnected_clusters',
    'ratio_wellconnected_clusters',
    'mixing_xi',
}

DISTR_STATS = {
    'degree',
    'concomp_sizes',

    'osub_degree',
    'o_deg',

    'c_size',
    'c_edges',
    'mincuts',
    'mixing_mus',
    'participation_coeffs',

    'o_participation_coeffs',
}


@click.command()
@click.option(
    '--input-network',
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
    ),
    help='Input network',
)
@click.option(
    '--input-clustering',
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
    ),
    help='Input clustering',
)
@click.option(
    '--output-folder',
    required=True,
    type=click.Path(
        file_okay=False,
    ),
    help='Ouput folder',
)
@click.option(
    '--overwrite',
    is_flag=True,
    help='Whether to overwrite existing data',
)
def compute_stats(input_network, input_clustering, output_folder, overwrite):
    # TODO: globally caching some intermediate results
    # TODO: refactor by abstracting the statistics
    # TODO: better profile memory and CPU usage

    # Prepare output folder
    dir_path = Path(output_folder)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Prepare agenda
    stats_to_compute = SCALAR_STATS | DISTR_STATS

    if not overwrite:
        existing_scalar_stats_file = dir_path / STATS_JSON_FILENAME
        existing_scalar_stats_dict = {}
        if existing_scalar_stats_file.is_file():
            with existing_scalar_stats_file.open('r') as f:
                existing_scalar_stats_dict = json.load(f)
        stats_to_compute -= set(existing_scalar_stats_dict.keys())

        existing_distr_stats_files = dir_path.glob('*.distribution')
        existing_distr_stats_names = [
            Path(existing_distr_stats_file).stem
            for existing_distr_stats_file in existing_distr_stats_files
        ]
        stats_to_compute -= set(existing_distr_stats_names)

    scalar_stats = {}
    distr_stats = {}

    # Start logging
    prepare_logging(dir_path, overwrite)
    job_start_time = time.perf_counter()

    # Read the network
    logging.info('Reading input network')
    start_time = time.perf_counter()
    elr = nk.graphio.EdgeListReader(
        '\t',
        0,
        continuous=False,
        directed=False,
    )
    graph = elr.read(input_network)
    graph.removeMultiEdges()
    graph.removeSelfLoops()
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    logging.info('Generating node mapping and ordering.')
    start_time = time.perf_counter()

    # Generate the node mapping
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

    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    logging.info('Generating cluster mapping and ordering.')
    start_time = time.perf_counter()

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

    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    logging.info('Generating outlier ordering.')
    start_time = time.perf_counter()

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

    # with open(dir_path / OUTLIERS_TSV_FILENAME, 'w') as idx_f:
    #     outlier_tsv_list = [
    #         [node]
    #         for node in outlier_nodes
    #     ]
    #     df = pd.DataFrame(outlier_tsv_list)
    #     df.to_csv(idx_f, sep='\t', header=False, index=False)

    o_subgraph = nk.graphtools.subgraphFromNodes(graph, outlier_nodes)
    c_subgraph = nk.graphtools.subgraphFromNodes(graph, clustered_nodes)

    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S1 - number of nodes
    logging.info('Stats - Number of nodes')
    start_time = time.perf_counter()
    if 'n_nodes' in stats_to_compute:
        n_nodes = compute_n_nodes(graph)
        scalar_stats['n_nodes'] = n_nodes
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S2 - number of edges
    logging.info('Stats - Number of edges')
    start_time = time.perf_counter()
    if 'n_edges' in stats_to_compute:
        n_edges = compute_n_edges(graph)
        scalar_stats['n_edges'] = n_edges
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S3 and S4 - number of connected components and connected components size distribution
    logging.info(
        'Stats - Number of connected components and connected components size distribution'
    )
    start_time = time.perf_counter()
    if 'n_concomp' in stats_to_compute or 'concomp_sizes' in stats_to_compute:
        n_concomp, concomp_sizes_distr = get_cc_stats(graph)
        scalar_stats['n_concomp'] = n_concomp
        distr_stats['concomp_sizes'] = concomp_sizes_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S5 - degree assortativity
    logging.info('Stats - Degree assortativity')
    start_time = time.perf_counter()
    if 'deg_assort' in stats_to_compute:
        deg_assort = compute_deg_assort(graph)
        scalar_stats['deg_assort'] = deg_assort
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S6 - Global Clustering Coefficient
    logging.info('Stats - Global clustering coefficient')
    start_time = time.perf_counter()
    if 'global_ccoeff' in stats_to_compute:
        global_ccoeff = compute_global_ccoeff(graph)
        scalar_stats['global_ccoeff'] = global_ccoeff
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S7 - Avg Local Clustering Coefficient
    logging.info('Stats - Avg Local clustering coefficient')
    start_time = time.perf_counter()
    if 'local_ccoeff' in stats_to_compute:
        local_ccoeff = compute_local_ccoeff(graph)
        scalar_stats['local_ccoeff'] = local_ccoeff
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S8 and S9 - degree distribution
    logging.info('Stats - Degree distribution')
    start_time = time.perf_counter()
    if 'degree' in stats_to_compute:
        deg_distr = compute_deg_distr(graph, node_order)
        distr_stats['degree'] = deg_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # TODO: S12?

    # S21 - diameter
    logging.info('Stats - Diameter')
    start_time = time.perf_counter()
    if 'diameter' in stats_to_compute:
        diameter = compute_diameter(graph)
        scalar_stats['diameter'] = diameter
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S23 - Jaccard similarity
    # TODO: this has not been implemented

    # S13 - number of outliers
    logging.info('Stats - Number of outliers')
    start_time = time.perf_counter()
    if 'n_onodes' in stats_to_compute:
        n_onodes = len(outlier_nodes)
        scalar_stats['n_onodes'] = n_onodes
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S14 number of edges among outliers nodes
    logging.info('Stats - Number of edges between outliers')
    start_time = time.perf_counter()
    if 'o_o_edges' in stats_to_compute:
        o_o_edges = o_subgraph.numberOfEdges()
        scalar_stats['o_o_edges'] = o_o_edges
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S15 number of edges between outlier and non-outlier nodes
    logging.info('Stats - Number of edges between outliers and non-outliers')
    start_time = time.perf_counter()
    if 'o_no_edges' in stats_to_compute:
        o_no_edges = n_edges - o_o_edges - c_subgraph.numberOfEdges()
        scalar_stats['o_no_edges'] = o_no_edges
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S16 degree distribution for the outlier node subgraph
    logging.info('Stats - Degree distribution for outlier node subgraph')
    start_time = time.perf_counter()
    if 'osub_degree' in stats_to_compute:
        osub_deg_distr = [
            o_subgraph.degree(u)
            for u in outlier_order
        ]
        distr_stats['osub_degree'] = osub_deg_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # TODO: any S?
    # outlier degree distribution
    logging.info('Stats - Degree distribution of the outliers')
    start_time = time.perf_counter()
    if 'o_deg' in stats_to_compute:
        o_deg_distr = [
            graph.degree(u)
            for u in outlier_order
        ]
        distr_stats['o_deg'] = o_deg_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S17 degree distribution for edges that connect outlier nodes to non-outlier nodes
    # TODO: Should this distribution include outlier-outlier edges?
    # TODO: this has not been implemented

    logging.info('Stats - Cluster stats')
    start_time = time.perf_counter()
    # TODO: break this into smaller parts
    if 'mincuts' in stats_to_compute \
            or 'n_clusters' in stats_to_compute \
            or 'c_size' in stats_to_compute \
            or 'c_edges' in stats_to_compute \
            or 'n_disconnects' in stats_to_compute \
            or 'ratio_disconnected_clusters' in stats_to_compute \
            or 'n_wellconnected_clusters' in stats_to_compute \
            or 'ratio_wellconnected_clusters' in stats_to_compute:
        cluster_stats = \
            compute_cluster_stats(
                input_network,
                input_clustering,
                cluster_mapping_dict_reversed,
                cluster_order,
            )
        cluster_stats.to_csv(dir_path / 'cluster_stats.csv', index=False)
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S19 and S20 - mininum cut size
    logging.info('Stats - Minimum cut size distribution')
    start_time = time.perf_counter()
    if 'mincuts' in stats_to_compute:
        mincuts_distr = cluster_stats['connectivity'].values
        distr_stats['mincuts'] = mincuts_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    logging.info('Stats - Number of clusters')
    start_time = time.perf_counter()
    if 'n_clusters' in stats_to_compute:
        n_clusters = len(cluster_stats)
        scalar_stats['n_clusters'] = n_clusters
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S10 and S11 - Cluster size distribution
    logging.info('Stats - Cluster size distribution')
    start_time = time.perf_counter()
    if 'c_size' in stats_to_compute or 'c_edges' in stats_to_compute:
        c_n_nodes_distr = cluster_stats['n'].values
        c_n_edges_distr = cluster_stats['m'].values
        distr_stats['c_size'] = c_n_nodes_distr
        distr_stats['c_edges'] = c_n_edges_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S18 - number of disconnected clusters
    logging.info('Stats - Connectivity')
    start_time = time.perf_counter()
    if 'n_disconnects' in stats_to_compute \
            or 'ratio_disconnected_clusters' in stats_to_compute \
            or 'n_wellconnected_clusters' in stats_to_compute \
            or 'ratio_wellconnected_clusters' in stats_to_compute:
        n_disconnected_clusters = \
            int((cluster_stats['connectivity'] < 1).sum())
        ratio_disconnected_clusters = \
            n_disconnected_clusters / n_clusters if n_clusters > 0 else 0
        n_wellconnected_clusters = \
            int((cluster_stats['connectivity_normalized_log10(n)'] > 1).sum())
        ratio_wellconnected_clusters = \
            n_wellconnected_clusters / n_clusters if n_clusters > 0 else 0
        scalar_stats['n_disconnects'] = n_disconnected_clusters
        scalar_stats['ratio_disconnected_clusters'] = ratio_disconnected_clusters
        scalar_stats['n_wellconnected_clusters'] = n_wellconnected_clusters
        scalar_stats['ratio_wellconnected_clusters'] = ratio_wellconnected_clusters
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S22 - mixing parameter
    logging.info('Stats - Mixing parameters')
    start_time = time.perf_counter()
    if 'mixing_mus' in stats_to_compute or 'mixing_xi' in stats_to_compute:
        mixing_mu_distr, mixing_xi = compute_mixing_params(
            graph,
            clustering_dict,
            node_mapping_dict_reversed,
            node_order,
        )
        distr_stats['mixing_mus'] = mixing_mu_distr
        scalar_stats['mixing_xi'] = mixing_xi
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # S24 - Participation coefficient distribution
    logging.info('Stats - Participation coefficients')
    start_time = time.perf_counter()
    if 'participation_coeffs' in stats_to_compute \
            or 'o_participation_coeffs' in stats_to_compute:
        participation_coeffs_distr, o_participation_coeffs_distr = \
            compute_participation_coeff_distr(
                graph,
                node_mapping_dict_reversed,
                clustering_dict,
                node_order,
                outlier_order
            )
        distr_stats['participation_coeffs'] = participation_coeffs_distr
        distr_stats['o_participation_coeffs'] = o_participation_coeffs_distr
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # Save scalar statistics
    logging.info('Saving scalar statistics')
    start_time = time.perf_counter()
    save_scalar_stats(dir_path, scalar_stats, overwrite)
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    # Save distribution statistics
    logging.info('Saving distribution statistics')
    start_time = time.perf_counter()
    save_distr_stats(dir_path, distr_stats, overwrite)
    logging.info(f'Time taken: {time.perf_counter() - start_time:.3f} seconds')

    logging.info(
        f'Total time taken: {time.perf_counter() - job_start_time:.3f} seconds'
    )

    # Save done file
    with open(dir_path / 'done', 'w') as f:
        f.write('done')


def prepare_logging(output_folder, is_overwrite):
    logging.basicConfig(
        filename=os.path.join(output_folder, STATS_LOG_FILENAME),
        filemode='w' if is_overwrite else 'a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)


def log_cpu_ram_usage(step_name):
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    logging.info(f'Step: {step_name} | CPU Usage: {cpu_percent}% | RAM Usage: {
                 ram_percent}% | Disk Usage: {disk_percent}')


def save_distr_stats(dir_path, distr_stats_dict, overwrite):
    distribution_arr = dir_path.glob('*.distribution')
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
    stats_dict = {}
    if stats_file.is_file():
        with stats_file.open('r') as f:
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


def compute_local_ccoeff(graph):
    return nk.globals.ClusteringCoefficient.sequentialAvgLocal(graph)


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
        # if u in node_mapping
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
        if n1 not in clustering_dict and n2 not in clustering_dict:
            continue
        elif n1 not in clustering_dict or n2 not in clustering_dict:
            out_degree[node1] += 1
            out_degree[node2] += 1
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
    totals = [in_degree[i] + out_degree[i] for i in graph.iterNodes()]
    xi = np.sum(outs) / np.sum(totals)

    # xi = np.sum(outs) / 2 / (graph.numberOfEdges())

    return mus, xi


def compute_diameter(graph):
    if graph.numberOfNodes() == 0:
        return 0
    connected_graph = \
        nk.components.ConnectedComponents.extractLargestConnectedComponent(
            graph, True)
    diam = nk.distance.Diameter(connected_graph, algo=1)
    diam.run()
    diameter = diam.getDiameter()
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

    participation_coeffs = {}
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

        participation_coeffs[node] = coeff

    participation_coeffs_distr = [
        participation_coeffs[v]
        for v in node_order
    ]

    o_participation_coeffs_distr = [
        participation_coeffs[v]
        for v in outlier_order
    ]

    return participation_coeffs_distr, o_participation_coeffs_distr


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
    edgelist_reader = nk.graphio.EdgeListReader('\t', 0)
    nk_graph = edgelist_reader.read(network_fp)

    global_graph = Graph(nk_graph, '')
    ms = [
        cluster.count_edges(global_graph)
        for cluster in clusters
    ]

    modularities = [
        global_graph.modularity_of(cluster)
        for cluster in clusters
    ]

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

    conductances = [
        cluster.conductance(global_graph)
        for cluster in clusters
    ]

    df = pd.DataFrame(
        list(
            zip(
                ids,
                ns,
                ms,
                modularities,
                mincuts,
                mincuts_normalized,
                conductances,
            )
        ),
        columns=[
            'cluster',
            'n',
            'm',
            'modularity',
            'connectivity',
            'connectivity_normalized_log10(n)',
            'conductance',
        ]
    )

    return df


if __name__ == '__main__':
    compute_stats()
