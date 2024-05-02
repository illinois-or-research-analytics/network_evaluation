# computing stats
import click

import networkit as nk
import numpy as np

@click.command()
@click.option("--input-network", required=True, type=click.Path(exists=True), help="Input network")
@click.option("--input-clustering", required=True, type=click.Path(exists=True), help="Input clustering")
@click.option("--node-ordering", required=False, type=click.Path(exists=True), help="Node ordering")
@click.option("--cluster-ordering", required=False, type=click.Path(exists=True), help="Cluster ordering")
@click.option("--output-folder", required=True, type=click.Path(), help="Ouput prefix")
def compute_stats(input_network, input_clustering, node_ordering, cluster_ordering, output_folder):
    """ input network and input clustering have no constraints
    files created inside output_folder where one file is created for generic stats and maybe more files
    for others
    """
    elr = nk.graphio.EdgeListReader('\t', 0, continuous=False, directed=False)
    graph = elr.read(input_network)

    num_nodes = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()

    output_data_file = f"{output_folder}/stats.dat"
    with open(output_data_file, "w") as f:
        f.write(f"number of nodes,scalar,{num_nodes}\n")
        f.write(f"number of edges,scalar,{num_edges}\n")

if __name__ == "__main__":
    compute_stats()
