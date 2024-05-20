# computing stats
import glob
import json
from pathlib import Path

import click

import networkit as nk
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

import cluster_stats as cls_stats


@click.command()
@click.option("--input-network", required=True, type=click.Path(exists=True), help="Input network")
@click.option("--input-clustering", required=True, type=click.Path(exists=True), help="Input clustering")
@click.option("--output-folder", required=True, type=click.Path(), help="Ouput folder")
@click.option("--overwrite", is_flag=True, help="Whether to overwrite existing data")
def compute_basic_stats(input_network, input_clustering, output_folder, overwrite):
    """ input network and input clustering have no constraints
    files created inside output_folder where one file is created for generic stats and maybe more files
    for others
    """
    # Read the network
    elr = nk.graphio.EdgeListReader('\t', 0, continuous=False, directed=False)
    graph = elr.read(input_network)
    graph.removeMultiEdges()
    graph.removeSelfLoops()

    # Prepare output folder
    dir_path = Path(output_folder)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Generate node ordering if external node ordering not provided!
    node_mapping_dict = elr.getNodeMap()
    node_mapping_dict_reversed = {
        v: int(k) for k, v in node_mapping_dict.items()}
    with open(str(dir_path)+"/node_ordering.idx", "w") as idx_f:
        for node in graph.iterNodes():
            idx_f.write(str(node_mapping_dict_reversed.get(node))+"\n")

    # Generate cluster ordering if external cluster ordering not provided!
    clustering_dict, cluster_ordering_dict = read_clustering(input_clustering)
    with open(str(dir_path)+"/cluster_ordering.idx", "w") as idx_f:
        for key, value in cluster_ordering_dict.items():
            idx_f.write(str(key)+"\n")

    # S1 and S2 - number of nodes and edges
    n_nodes = graph.numberOfNodes()
    n_edges = graph.numberOfEdges()

    # S3 and S4 - number of connected components and connected components size distribution
    n_concomp, concomp_sizes_distr = get_cc_stats(graph)

    # S5 degree assortativity
    # convert from NetworKit.Graph to networkx.Graph
    nx_graph = nk.nxadapter.nk2nx(graph)
    deg_assort = round(nx.degree_assortativity_coefficient(nx_graph), 5)

    # S6 Global Clustering Coefficient
    global_ccoeff = round(
        nk.globals.ClusteringCoefficient.exactGlobal(graph), 5)

    # S8 and S9 - degree distribution, degree sequence
    deg_distr = get_degree_distr(graph)

    # S10 and S11 - Cluster size distribution
    cluster_size_distr = get_cluster_size_distr(clustering_dict)

    """outlier nodes stats"""
    # S13 number of outliers
    outlier_nodes, clustered_nodes = get_outliers(
        graph, node_mapping_dict, clustering_dict)
    n_onodes = len(outlier_nodes)
    with open(str(dir_path)+"/outlier_ordering.idx", "w") as idx_f:
        for node in outlier_nodes:
            idx_f.write(str(node_mapping_dict_reversed.get(node))+"\n")
    with open(str(dir_path)+"/outliers.tsv", "w") as idx_f:
        for node in outlier_nodes:
            idx_f.write(str(node)+"\n")

    """ ourlier edge stats!"""
    # S14 number of edges among outliers nodes
    o_subgraph = nk.graphtools.subgraphFromNodes(graph, outlier_nodes)
    o_o_edges = o_subgraph.numberOfEdges()

    # S15 number of edges between outlier and non-outlier nodes
    clustered_subgraph = nk.graphtools.subgraphFromNodes(
        graph, clustered_nodes)
    o_no_edges = n_edges - o_o_edges - clustered_subgraph.numberOfEdges()

    # S16 degree distribution for the outlier node subgraph
    osub_deg_distr = [o_subgraph.degree(u) for u in outlier_nodes]

    # outlier degree distribution
    o_deg_distr = [graph.degree(u) for u in outlier_nodes]

    # S17 degree distribution for edges that connect outlier nodes to non-outlier nodes
    # Should this distribution include outlier-outlier edges?
    """Cluster Statistics!"""
    # Getting cluster statistics
    cluster_stats_path = str(dir_path) + "/cluster_stats.csv"
    cls_stats.main(input=input_network, existing_clustering=input_clustering,
                   resolution=-1, universal_before="", output=cluster_stats_path)
    # Reading the saved cluster statistics
    cluster_stats = pd.read_csv(cluster_stats_path)
    cluster_stats = cluster_stats.drop(cluster_stats.index[-1])

    # S18 number of disconnected clusters
    disconnected_clusters = (
        cluster_stats['connectivity_normalized_log10(n)'] < 1).sum()

    # S19 and S20 mininum cut size distribution - mincut sequence
    mincuts_distr = cluster_stats['connectivity'].values

    # S21 diameter
    diameter = compute_diameter(graph)

    # S22 mixiting parameter
    mixing_param = compute_mixing_param(graph, clustering_dict, node_mapping_dict_reversed)

    # S23 Jaccard similarity

    # S24 Participation coefficient distribution
    participation_dict = get_participation_coeffs(
        graph, clustering_dict, node_mapping_dict_reversed)
    print("Started computing participation coefficients! ")
    participation_coeffs = []
    outlier_participation_coeffs = {}
    for node, participation in participation_dict.items():
        deg_of_node = sum(list(participation.values()))
        coeff = 1
        if deg_of_node > 0:
            coeff -= np.sum([(deg_i/deg_of_node) **
                            2 for deg_i in list(participation.values())])
            if -1 in participation.keys():
                coeff += (participation.get(-1)/deg_of_node)**2
                coeff -= participation.get(-1)*((1/deg_of_node)**2)
            coeff = round(coeff, 5)
        participation_coeffs.append(coeff)
        if node in outlier_nodes:
            outlier_participation_coeffs[node] = coeff
    o_participation_coeffs_distr = [
        outlier_participation_coeffs.get(v) for v in outlier_nodes]

    # Save scalar statistics
    stats_dict = {}
    stats_file = Path(output_folder) / "stats.json"
    file_rw_bit = "w"
    if stats_file.is_file():
        file_rw_bit = "r"
        with stats_file.open(file_rw_bit) as f:
            stats_dict = json.load(f)

    if "n_nodes" not in stats_dict or overwrite:
        stats_dict["n_nodes"] = n_nodes
    if "n_edges" not in stats_dict or overwrite:
        stats_dict["n_edges"] = n_edges
    if "n_onodes" not in stats_dict or overwrite:
        stats_dict["n_onodes"] = n_onodes
    if "o_o_edges" not in stats_dict or overwrite:
        stats_dict["o_o_edges"] = o_o_edges
    if "o_no_edges" not in stats_dict or overwrite:
        stats_dict["o_no_edges"] = o_no_edges
    if "n_concomp" not in stats_dict or overwrite:
        stats_dict["n_concomp"] = n_concomp
    if "deg_assort" not in stats_dict or overwrite:
        stats_dict["deg_assort"] = deg_assort
    if "global_ccoeff" not in stats_dict or overwrite:
        stats_dict["global_ccoeff"] = global_ccoeff
    if "n_disconnects" not in stats_dict or overwrite:
        stats_dict["n_disconnects"] = int(disconnected_clusters)
    if "mixing_param" not in stats_dict or overwrite:
        stats_dict["mixing_param"] = mixing_param
    if "diameter" not in stats_dict or overwrite:
        stats_dict["diameter"] = diameter

    with stats_file.open("w") as f:
        json.dump(stats_dict, f, indent=4)

    # save distribution statistics
    distr_stats_dict = {}
    distr_stats_dict['degree'] = deg_distr
    distr_stats_dict['concomp_sizes'] = concomp_sizes_distr
    distr_stats_dict['osub_degree'] = osub_deg_distr
    distr_stats_dict['o_deg'] = o_deg_distr
    distr_stats_dict['c_size'] = cluster_size_distr
    distr_stats_dict['mincuts'] = mincuts_distr
    distr_stats_dict['participation_coeffs'] = participation_coeffs
    distr_stats_dict['o_participation_coeffs'] = o_participation_coeffs_distr

    distr_stats = ['degree', 'concomp_sizes', 'osub_degree', 'o_deg',
                   'c_size', 'mincuts', 'participation_coeffs', 'o_participation_coeffs']
    distribution_arr = glob.glob(f"{dir_path}/*.distribution")
    distribution_name_arr = []
    for current_distribution_file in distribution_arr:
        distribution_name = Path(current_distribution_file).stem
        distribution_name_arr.append(distribution_name)

    for distr_stat in distr_stats:
        if f"{distr_stat}.distribution" not in distribution_name_arr or overwrite:
            with open(str(dir_path)+f"/{distr_stat}.distribution", "w") as distr_f:
                for item in distr_stats_dict.get(distr_stat):
                    distr_f.write(str(item)+"\n")


def read_clustering(filepath):
    """Read clustering and return the cluster dict and cluster order mapping"""
    cluster_df = pd.read_csv(filepath, sep="\t", header=None, names=[
                             "node_id", "cluster_name"])
    unique_values = cluster_df["cluster_name"].unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)
    clustering_dict = dict(
        zip(cluster_df['node_id'], cluster_df['cluster_id']))
    return clustering_dict, value_map


def get_outliers(graph, node_mapping, clustering_dict):
    clustered_nodes = [node_mapping.get(str(u))
                       for u in clustering_dict.keys()]
    nodes_set = set()
    for u in graph.iterNodes():
        nodes_set.add(u)
    outlier_nodes = nodes_set.difference(clustered_nodes)
    return outlier_nodes, clustered_nodes


def get_degree_distr(graph):
    return [graph.degree(v) for v in graph.iterNodes()]


def get_cc_stats(graph):
    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    num_cc = cc.numberOfComponents()
    cc_size_distribution = cc.getComponentSizes()
    return num_cc, cc_size_distribution.values()


def get_cluster_size_distr(clustering_dict):
    cluster_size_dict = {}
    for cluster in clustering_dict.values():
        cluster_size_dict[cluster] = cluster_size_dict.get(cluster, 0) + 1
    cluster_size_distr = []
    for i in range(len(cluster_size_dict.keys())):
        cluster_size_distr.append(cluster_size_dict.get(i))
    return cluster_size_distr


def compute_mixing_param(net, clustering_dict, node_mapping_dict_reversed):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    clustered_keys = clustering_dict.keys()
    for node1, node2 in net.iterEdges():
        n1 = node_mapping_dict_reversed.get(node1)
        n2 = node_mapping_dict_reversed.get(node2)
        if n1 in clustered_keys and n2 in clustered_keys:
            if clustering_dict[n1] == clustering_dict[n2]: # nodes are co-clustered
                in_degree[n1] += 1
                in_degree[n2] += 1
        else:
            out_degree[n1] += 1
            out_degree[n2] += 1
    mus = [out_degree[i]/(out_degree[i]+in_degree[i]) if (out_degree[i]+in_degree[i]) != 0 else 0 for i in net.iterNodes()]
    mixing_param = np.mean(mus)
    return round(mixing_param, 4)

def compute_diameter(graph):
    connected_graph = nk.components.ConnectedComponents.extractLargestConnectedComponent(graph, True)
    diam = nk.distance.Diameter(connected_graph,algo=1)
    diam.run()
    diameter = diam.getDiameter()
    return diameter[0]

def get_participation_coeffs(graph, clustering_dict, node_mapping_dict_reversed):
    participation_dict = defaultdict(dict)
    for v in graph.iterNodes():
        for neighbor in graph.iterNeighbors(v):
            neighbor_cluster = clustering_dict.get(
                node_mapping_dict_reversed.get(neighbor))
            if neighbor_cluster is None:
                neighbor_cluster = -1
            node_participation_dict = participation_dict[v]
            if neighbor_cluster in node_participation_dict.keys():
                node_participation_dict[neighbor_cluster] = node_participation_dict.get(
                    neighbor_cluster) + 1
            else:
                node_participation_dict[neighbor_cluster] = 1
        if graph.isIsolated(v):
            participation_dict[v] = {-1: 0}

    return participation_dict


if __name__ == "__main__":
    compute_basic_stats()
