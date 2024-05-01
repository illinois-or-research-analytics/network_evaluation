import typer
import networkit as nk
import pandas as pd
import os
import json
import logging
import matplotlib.pyplot as plt
import argparse
import numpy as np 
from collections import defaultdict

def get_isolated_vertices(graph, clustering_dict, node_mapping):
    numerical_to_string_mapping = {v: int(k) for k, v in node_mapping.items()}
    isolated_vertices = []
    degrees_dict = {}
    degrees_list = []
    check_participation = False
    participation_dict = defaultdict(dict)
    if len(clustering_dict)>1:
        check_participation = True
    for v in graph.iterNodes():
        isolated = False
        deg = graph.degree(v)
        if graph.isIsolated(v):
            isolated_vertices.append(v)
            isolated = True
        
        if check_participation:
            for neighbor in graph.iterNeighbors(v):
                neighbor_cluster = clustering_dict.get(numerical_to_string_mapping.get(neighbor))
                if neighbor_cluster is None:
                    neighbor_cluster = -1
                node_participation_dict = participation_dict[numerical_to_string_mapping.get(v)]
                if neighbor_cluster in node_participation_dict.keys():
                    node_participation_dict[neighbor_cluster] = node_participation_dict.get(neighbor_cluster) + 1
                else:
                    node_participation_dict[neighbor_cluster] = 1
            if isolated:
                participation_dict[numerical_to_string_mapping.get(v)] = {-1:0}
                
        degrees_list.append(deg)
        degrees_dict[v] = deg
    return isolated_vertices, degrees_list, degrees_dict, participation_dict

def get_graph_stats(graph, outlier_nodes, stage, node_mapping, clustering_dict):
    print("Number of self loops : ",graph.numberOfSelfLoops())
    
    stats = {}
    num_vertices = graph.numberOfNodes()
    num_edges_read = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges_read)
    stats[f'num_vertices_{stage}'] = num_vertices
    stats[f'num_edges_{stage}'] = num_edges_read

    graph.removeMultiEdges()
    graph.removeSelfLoops()
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Number of parallel/multiedges : " , (num_edges_read-num_edges))
    print("Num edges after removing self-loops and duplicate parallel edges: ", num_edges)
    stats[f'num_vertices_cleaned_{stage}'] = num_vertices
    stats[f'num_edges_cleaned_{stage}'] = num_edges
    isolated_vertices, degrees_list, degrees_dict, participation_dict = get_isolated_vertices(graph, clustering_dict, node_mapping)
    stats[f'num_isolated_{stage}'] = len(isolated_vertices)

    # dd = sorted(nk.centrality.DegreeCentrality(graph).run().scores(), reverse=True)
    # degrees, numberOfNodes = np.unique(dd, return_counts=True)
    degrees_list = np.array(degrees_list)
    degrees, counts = np.unique(degrees_list, return_counts=True)
    if degrees.min()>1:
        degrees = np.append(degrees,1)
        counts = np.append(counts,0)
    fig = plt.figure()
    plt.title(f"Degree distribution - {stage}")
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    plt.plot(degrees, counts)

    degree_stats = [np.min(degrees_list),np.percentile(degrees_list, 25),np.median(degrees_list), np.percentile(degrees_list, 75),np.max(degrees_list),np.average(degrees_list)]
    stats[f'degree_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in degree_stats]
    outlier_degrees = []
    print("Started outlier degree statistics!")
    outlier_degrees_dict = {}
    mapped_outlier_nodes = set()
    for node in outlier_nodes:
        mapped_outlier_nodes.add(node_mapping.get(str(node)))
    print("Got mapped outlier nodes!")
    # if(len(outlier_nodes)>0):
    #     outlier_node_set = set(outlier_nodes)
    #     outlier_edges_new = 0
    #     for node in participation_dict.keys():
    #         if node in outlier_node_set  and -1 in participation_dict.get(node).keys():
    #             outlier_edges_new += participation_dict.get(node).get(-1)
    #     print("New logic to compute outlier-outlier edges !", outlier_edges_new)
    if len(outlier_nodes)>0:
        outlier_edges = 0
        for edge in graph.iterEdges():
            if edge[0] in mapped_outlier_nodes:
                deg = degrees_dict.get(edge[0])
                outlier_degrees_dict[edge[0]] = deg
                if edge[1] in mapped_outlier_nodes:
                    outlier_edges += 1
        print("Processed all edges")
        stats[f'outlier_edges_{stage}'] = outlier_edges
        outlier_degrees = list(outlier_degrees_dict.values())
        outlier_degree_stats = [np.min(outlier_degrees),np.percentile(outlier_degrees, 25),np.median(outlier_degrees),np.percentile(outlier_degrees, 75),np.max(outlier_degrees), np.average(outlier_degrees)]
        stats[f'outlier_degree_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in outlier_degree_stats]
        isolated_outlier_vertices = set(isolated_vertices) & mapped_outlier_nodes
        stats[f'isolated_outlier_nodes_{stage}'] = len(isolated_outlier_vertices)
    print("Outlier degree statistics done")

    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    stats[f'Num_connected_components_{stage}'] = cc.numberOfComponents()
    stats[f'Size_connected_components_{stage}'] = cc.getComponentSizes()

    #Computing participation coefficients!
    print("Started computing participation coefficients! ")
    participation_coeffs = {}
    outlier_nodes_set = set(outlier_nodes)
    print("Difference in sets : ", len(mapped_outlier_nodes - outlier_nodes_set))
    clustered_participation_coeffs = {}
    outlier_participation_coeffs = {}
    for node,participation in participation_dict.items():
        deg_of_node = sum(list(participation.values()))
        coeff = 1
        if deg_of_node > 0 :
            # participation_values = [participation_value for key, participation_value in participation.items() if key != -1]
            coeff -= np.sum([(deg_i/deg_of_node)**2 for deg_i in list(participation.values())])

        participation_coeffs[node] = coeff

        if node in outlier_nodes_set:
            outlier_participation_coeffs[node] = coeff
        else:
            clustered_participation_coeffs[node] = coeff

    # print(len(outlier_participation_coeffs), len(clustered_participation_coeffs), len(participation_coeffs))
    outlier_coeffs = list(outlier_participation_coeffs.values())
    outlier_coeff_stats = [np.min(outlier_coeffs),np.percentile(outlier_coeffs, 25),np.median(outlier_coeffs),np.percentile(outlier_coeffs, 75),np.max(outlier_coeffs), np.average(outlier_coeffs)]
    
    clustered_coeffs = list(clustered_participation_coeffs.values())
    clustered_coeff_stats = [np.min(clustered_coeffs),np.percentile(clustered_coeffs, 25),np.median(clustered_coeffs),np.percentile(clustered_coeffs, 75),np.max(clustered_coeffs), np.average(clustered_coeffs)]

    coeffs = list(participation_coeffs.values())
    coeffs_dist_stats = [np.min(coeffs),np.percentile(coeffs, 25),np.median(coeffs),np.percentile(coeffs, 75),np.max(coeffs), np.average(coeffs)]
    
    stats[f'participationCoeff_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in coeffs_dist_stats]
    stats[f'outlierParticipationCoeff_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in outlier_coeff_stats]
    stats[f'clusteredParticipationCoeff_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in clustered_coeff_stats]
    
    print("Completed computing participation coefficients!")

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    return stats_df, fig, participation_coeffs,participation_dict

def read_graph(filepath):
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0, directed=False, continuous=False)
    nk_graph = edgelist_reader.read(filepath)
    node_mapping = edgelist_reader.getNodeMap()
    return nk_graph, node_mapping

def read_clustering(filepath):
    cluster_df = pd.read_csv(filepath, sep="\t", header=None, names=["node_id", "cluster_name"])
    unique_values = cluster_df["cluster_name"].unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)
    return cluster_df[['node_id', 'cluster_id']]


def main(filepath, outlier_nodes, stage, clustering_filepath):
    graph, node_mapping = read_graph(filepath)
    clustering_dict = {}
    if len(clustering_filepath)>1:
        cluster_df = read_clustering(clustering_filepath)
        clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
    if (len(outlier_nodes)==0 and len(clustering_dict.keys())>0):
        print("Getting outlier nodes from clustering!")
        node_mapping_reversed = {v: int(k) for k, v in node_mapping.items()}
        clustered_nodes_id_org = cluster_df['node_id'].to_numpy()
        nodes_set = set()
        for u in graph.iterNodes():
            nodes_set.add(node_mapping_reversed.get(u))
        unclustered_nodes = nodes_set.difference(clustered_nodes_id_org)
        outlier_nodes.extend(unclustered_nodes)
        print("Number of outlier nodes found : ", len(outlier_nodes))

    stats_df, fig, participation_coeffs,participation_dict = get_graph_stats(graph, outlier_nodes, stage, node_mapping, clustering_dict)
    return stats_df, fig, participation_coeffs,participation_dict

if __name__ == "__main__":
    main()
