# Network Evaluation
This repository contains scripts for network evaluation including
- Computing clustering accuracy of an estimated clustering to the ground-truth clustering of a given network
- Computing network level statistics
- Computing cluster level statistics

## Clustering Accuracy
This section describes the command and usage for computing the clustering accuracy of a disjoint clustering of a network given a disjoint ground-truth clustering. The input network and clustering files are  expected to be in a comma separated format without headers.

```
python <git root>/commdet_acc/compute_cd_accuracy.py --input-network <FILE> --gt-clustering <FILE> --est-clustering <FILE> --output-prefix <FILE PATH PREFIX> --num_processors <INT>
```
## Statistics
### Network Statistics
This section describes the command and usage for computing network level statistics for a given network. The input network file is  expected to be in a comma separated format without headers.
```
python <git root>/network_stats/compute_network_stats.py --network <FILE> --gt-clustering <FILE> --outdir <FILE PATH PREFIX> --overwrite <BOOL>
```

|output field name | explanation|
|---|---|
|n_nodes| number of nodes|
|n_edges| number of edges|
|n_concomp| number of connected components|
|deg_assort| degree assortativity|
|global_ccoeff| global clustering coefficient|
|local_ccoeff| average of local clustering coefficients|
|diameter| pseudo-diameter (diameter of the largest connected component)|
|degree| degree sequence|
|concomp_sizes| sizes of the connected components|

### Clustering Statistics
This section describes the command and usage for computing cluster level statistics for a given network and a clustering. The input network and clustering files are  expected to be in a comma separated format without headers.
```
python <git root>/network_stats/compute_cluster_stats.py --network <FILE> --gt-clustering <FILE>  --community <FILE> --outdir <FILE PATH PREFIX>
```
| output field name | explanation |
| --- | --- |
|global_n| number of nodes|
|global_m| number of edges|
|n_outliers| number of outliers (unclustered nodes)|
|node_coverage| proportion of outliers|
|mixing_parameter| mixing parameters (proportion of boundary edges) of nodes (each outlier has  a mixing parameter of 1.0)|
|n| number of nodes internal to a cluster|
|m| number of edges internal to a cluster|
|c| number of edges on the boundary of a cluster (one end inside, one end outside)|
|conductance| conductances of clusters|
|degree_density| degree densities (\|E(S)\|/\|S\|) of clusters |
|edge_density| edge densities (\|E(S)\|/C(\|S\|, 2)) of clusters|
|mincut| sizes of minimum edge-cut of clusters|
|modularity| modularity scores of clusters|
|n_clusters| number of clusters|
|n_singleton_clusters| number of singleton clusters (with n = 1)|
|n_disconnected_clusters| number of disconnected clusters (with n > 1, mincut = 0)|
|n_connected_clusters| number of connected clusters (with n > 1, mincut > 0)|
|n_wellconnected_clusters| number of well-connected clusters (with n > 1, mincut > log10(n))|
