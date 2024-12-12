import json
import argparse
from pathlib import Path
import logging

import graph_tool.all as gt
import numpy as np
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Compute excess edges in a graph.'
)
parser.add_argument(
    '--input',
    type=str,
    help='Path to the network file',
)
# parser.add_argument(
#     '--output',
#     type=str,
#     help='Path to the output JSON file',
# )

args = parser.parse_args()

network_fp = Path(args.input)
# output_fp = Path(args.output)

assert network_fp.exists(), f'File not found: {network_fp}'
assert network_fp.is_file(), f'Not a file: {network_fp}'
# assert output_fp.parent.exists(), f'Directory not found: {output_fp.parent}'
# assert output_fp.parent.is_dir(), f'Not a directory: {output_fp.parent}'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info(f'Input: {network_fp}')
# logger.info(f'Output: {output_fp}')

G = gt.load_graph_from_csv(
    str(network_fp),
    directed=False,
    csv_options={'delimiter': '\t'},
)

# n_edges = G.num_edges()

gt.remove_parallel_edges(G)
# n_parallel_edges = n_edges - G.num_edges()

gt.remove_self_loops(G)
# n_self_loops = n_edges - n_parallel_edges - G.num_edges()

# logger.info(f'n_edges: {n_edges}')
# logger.info(f'n_parallel_edges: {n_parallel_edges}')
# logger.info(f'n_self_loops: {n_self_loops}')
# logger.info(f'ratio_parallel_edges: {n_parallel_edges / n_edges}')
# logger.info(f'ratio_self_loops: {n_self_loops / n_edges}')

# with open(output_fp, 'w') as f:
#     json.dump({
#         'n_edges': n_edges,
#         'n_parallel_edges': n_parallel_edges,
#         'n_self_loops': n_self_loops,
#         'ratio_parallel_edges': n_parallel_edges / n_edges,
#         'ratio_self_loops': n_self_loops / n_edges,
#     }, f, indent=4)

# Degree assortativity
# assort, _ = gt.assortativity(G, 'total')
deg_assort, _ = gt.scalar_assortativity(G, 'total')
logger.info(f'Scalar assortativity: {deg_assort}')

# Mean k-core value
core = gt.kcore_decomposition(G)
mean_k_core = np.mean([
    core[v] for v in G.vertices()
])
logger.info(f'Mean k-core value: {mean_k_core}')

# Local clustering coefficient
local_ccoeffs = gt.local_clustering(G)
local_ccoeff = np.mean([
    local_ccoeffs[v] for v in G.vertices()
])
logger.info(f'Mean local clustering coefficient: {local_ccoeff}')

# Global clustering coefficient
global_ccoeff, _ = gt.global_clustering(G)
logger.info(f'Global clustering coefficient: {global_ccoeff}')

# Leading eigenvalue of the adjacency matrix
largest_eigval_A, _ = gt.eigenvector(G)
logger.info(f'Leading eigenvalue of the adjacency matrix: {largest_eigval_A}')

# Leading eigenvalue of the Hashimoto matrix
H_mtx = gt.hashimoto(G)
eigvals_H = la.eigs(H_mtx, k=1, return_eigenvectors=False, which='LR')
largest_eigval_H = eigvals_H[0].real
logger.info(f'Leading eigenvalue of the Hashimoto matrix: {largest_eigval_H}')

# Characteristic time of a random walk
# Get the largest connected component
largest_cc = gt.extract_largest_component(G)
# Compute the transition matrix
T = gt.transition(largest_cc)
# Compute the 2nd largest eigenvalues of the transition matrix
eigvals_T = la.eigs(T, k=2, return_eigenvectors=False, which='LR')
# Compute the characteristic time
tau = -np.log(np.abs(eigvals_T[0].real))
logger.info(f'Characteristic time of a random walk: {tau}')

# Psuedo-diameter
pseudo_diameter, *_ = gt.pseudo_diameter(G)
logger.info(f'Pseudo-diameter: {pseudo_diameter}')

# Node percolation profile (targeted removal)
vertices = sorted([v for v in G.vertices()], key=lambda v: v.out_degree())
sizes, _ = gt.vertex_percolation(G, vertices)
Rt = np.mean(sizes) / G.num_vertices()
logger.info(f'Node percolation profile (targeted removal): {Rt}')

# Node percolation profile (random removal)
n_trials = 100
Rr = 0.0
vertices = [v for v in G.vertices()]
for i in range(n_trials):
    # node_orderings = np.random.permutation(n_nodes)
    np.random.shuffle(vertices)
    sizes2, _ = gt.vertex_percolation(G, vertices)
    Rr += np.mean(sizes2) / G.num_vertices() / n_trials
logger.info(f'Node percolation profile (random removal): {Rr}')

# Fraction of vertices in the giant component
giant_cc = gt.extract_largest_component(G)
giant_frac = giant_cc.num_vertices() / G.num_vertices()
logger.info(f'Fraction of vertices in the giant component: {giant_frac}')
