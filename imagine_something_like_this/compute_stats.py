# computing stats
import glob
import json
from pathlib import Path

import click

import networkit as nk
import numpy as np
import pandas as pd
import networkx as nx

@click.command()
@click.option("--input-network", required=True, type=click.Path(exists=True), help="Input network")
@click.option("--input-clustering", required=True, type=click.Path(exists=True), help="Input clustering")
@click.option("--node-ordering", required=False, type=click.Path(exists=True), help="Node ordering")
@click.option("--cluster-ordering", required=False, type=click.Path(exists=True), help="Cluster ordering")
@click.option("--output-json", required=True, type=click.Path(), help="Ouput json file")
@click.option("--overwrite", is_flag=True, help="Whether to overwrite existing data")
def compute_basic_stats(input_network, input_clustering, node_ordering, cluster_ordering, output_json, overwrite):
    """ input network and input clustering have no constraints
    files created inside output_folder where one file is created for generic stats and maybe more files
    for others
    """
    #Read the network
    elr = nk.graphio.EdgeListReader('\t', 0, continuous=False, directed=False)
    graph = elr.read(input_network)

    #Generate node ordering if external node ordering not provided!
    if node_ordering is None:
        node_ordering_dict = elr.getNodeMap()
        node_ordering_dict_reversed = {v: int(k) for k, v in node_ordering_dict.items()}
        dir_path = Path(output_json).parent
        with open(str(dir_path)+"/node_ordering.idx","w") as idx_f:
            for node in graph.iterNodes():
                idx_f.write(str(node_ordering_dict_reversed.get(node))+"\n")

    #Generate cluster ordering if external cluster ordering not provided!
    clustering_dict, cluster_ordering_dict = read_clustering(input_clustering)
    if cluster_ordering is None:
        with open(str(dir_path)+"/cluster_ordering.idx","w") as idx_f:
            for key,value in cluster_ordering_dict.items():
                idx_f.write(str(key)+"\n")

    #S1 and S2 - number of nodes and edges
    n_nodes = graph.numberOfNodes()
    n_edges = graph.numberOfEdges()

    #S3 and S4 - number of connected components and connected components size distribution
    n_concomp , concomp_sizes_distr = get_cc_stats(graph)

    #S5 degree assortativity
    nx_graph = nk.nxadapter.nk2nx(graph) # convert from NetworKit.Graph to networkx.Graph
    deg_assort = round(nx.degree_assortativity_coefficient(nx_graph),5)

    #S6 Global Clustering Coefficient
    global_ccoeff = round(nk.globals.ClusteringCoefficient.exactGlobal(graph),5)

    #S8 and S9 - degree distribution, degree sequence 
    deg_distr = get_degree_distr(graph)

    #S10 and S11 - Cluster size distribution
    cluster_size_distr = get_cluster_size_distr(clustering_dict)

    """outlier nodes stats"""
    #S13 number of outliers
    outlier_nodes, clustered_nodes = get_outliers(graph, node_ordering_dict , clustering_dict)
    n_onodes = len(outlier_nodes)
    with open(str(dir_path)+"/outlier_ordering.idx","w") as idx_f:
            for node in outlier_nodes:
                idx_f.write(str(node_ordering_dict_reversed.get(node))+"\n")
    with open(str(dir_path)+"/outliers.tsv","w") as idx_f:
            for node in outlier_nodes:
                idx_f.write(str(node)+"\n")

    """ ourlier edge stats!"""
    #S14 number of edges among outliers nodes
    o_subgraph = nk.graphtools.subgraphFromNodes(graph, outlier_nodes)
    o_o_edges = o_subgraph.numberOfEdges() 

    #S15 number of edges between outlier and non-outlier nodes
    clustered_subgraph = nk.graphtools.subgraphFromNodes(graph, clustered_nodes)
    o_no_edges = n_edges - o_o_edges - clustered_subgraph.numberOfEdges() 

    #S16 degree distribution for the outlier node subgraph
    osub_deg_distr = [o_subgraph.degree(u) for u in outlier_nodes]

    # outlier degree distribution
    o_deg_distr = [graph.degree(u) for u in outlier_nodes]

    #S17 degree distribution for edges that connect outlier nodes to non-outlier nodes
    # Should this distribution include outlier-outlier edges?

    #S18 number of disconnected clusters
    

    #S19 and S20 mininum cut size distribution - mincut sequence

    #S21 diameter

    #S22 mixiting time

    #S23 Jaccard similarity

    #S24 Participation coefficient distribution

    
    #Save scalar statistics
    stats_dict = {}
    stats_file = Path(output_json)
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

    with stats_file.open("w") as f:
        json.dump(stats_dict, f, indent=4)

    #save distribution statistics
    distr_stats_dict = {}
    distr_stats_dict['degree'] = deg_distr
    distr_stats_dict['concomp_sizes'] = concomp_sizes_distr
    distr_stats_dict['osub_degree'] = osub_deg_distr
    distr_stats_dict['o_deg_distr'] = o_deg_distr
    distr_stats_dict['c_size_distr'] = cluster_size_distr
    distr_stats = ['degree', 'concomp_sizes','osub_degree','o_deg_distr','c_size_distr']
    dir_path = Path(output_json).parent
    distribution_arr = glob.glob(f"{dir_path}/*.distribution")
    distribution_name_arr = []
    for current_distribution_file in distribution_arr:
        distribution_name = Path(current_distribution_file).stem
        distribution_name_arr.append(distribution_name)

    for distr_stat in distr_stats:
        if f"{distr_stat}.distribution" not in distribution_name_arr or overwrite:
            with open(str(dir_path)+f"/{distr_stat}.distribution","w") as distr_f:
                for item in distr_stats_dict.get(distr_stat):
                    distr_f.write(str(item)+"\n")


def read_clustering(filepath):
    """Read clustering and return the cluster dict and cluster order mapping"""
    cluster_df = pd.read_csv(filepath, sep="\t", header=None, names=["node_id", "cluster_name"])
    unique_values = cluster_df["cluster_name"].unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)
    clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
    return clustering_dict, value_map

def get_outliers(graph, node_mapping, clustering_dict):
    clustered_nodes = [node_mapping.get(str(u)) for u in clustering_dict.keys()]
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

if __name__ == "__main__":
    compute_basic_stats()
