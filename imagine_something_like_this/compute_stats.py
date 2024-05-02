# computing stats
import json
from pathlib import Path

import click

import networkit as nk
import numpy as np

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
    elr = nk.graphio.EdgeListReader('\t', 0, continuous=False, directed=False)
    graph = elr.read(input_network)

    num_nodes = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()

    stats_dict = {}
    stats_file = Path(output_json)
    file_rw_bit = "w"
    if stats_file.is_file():
        file_rw_bit = "r"
    with stats_file.open(file_rw_bit) as f:
        stats_dict = json.load(f)

    if "num_nodes" not in stats_dict or overwrite:
        stats_dict["num_nodes"] = num_nodes
    if "num_edges" not in stats_dict or overwrite:
        stats_dict["num_edges"] = num_edges

    with stats_file.open("w") as f:
        json.dump(stats_dict, f, indent=4)


if __name__ == "__main__":
    compute_basic_stats()
