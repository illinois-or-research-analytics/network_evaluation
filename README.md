# Network Evaluation
This repository contains scripts for network evaluation including
- Computing clustering accuracy of an estimated clustering to the ground-truth clustering of a given network
- Computing network level statistics
- Computing cluster level statistics

**Note** We call nodes that are not assigned to any cluster or are in clusters of size 1 as outliers/unclustered nodes (thus, there will never be a cluster with size 1). The clustering accuracy scores are computed by treating each outlier as a singleton cluster.

## Clustering Accuracy
This section describes the command and usage for computing the clustering accuracy of a disjoint clustering of a network given a disjoint ground-truth clustering. The input network and clustering files are expected to be in a comma separated format without headers.

```
python <git root>/commdet_acc/compute_cd_accuracy.py --input-network <FILE> --gt-clustering <FILE> --est-clustering <FILE> --output-prefix <FILE PATH PREFIX> --num_processors <INT>
```
## Statistics
### Network Statistics
This section describes the command and usage for computing network level statistics for a given network. The input network file is expected to be in a comma separated format without headers.
```
python <git root>/network_stats/compute_network_stats.py --network <FILE> --gt-clustering <FILE> --outdir <FILE PATH PREFIX> --overwrite <BOOL>
```

The output contains a `stats.json` file with multiple fields and TXT files, each corresponding to a different statistic. The description of the fields and the files are as follows.

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

Additionally, there is a `node.idx` file that contains the original node ids. Each line in this file corresponds to the same line in the TXT files.

### Clustering Statistics
This section describes the command and usage for computing cluster level statistics for a given network and a clustering. The input network and clustering files are expected to be in a comma separated format without headers.
```
python <git root>/network_stats/compute_cluster_stats.py --network <FILE> --community <FILE> --outdir <FILE PATH PREFIX>
```

The output contains multiple TXT files, each corresponding to a different statistic. They are named as follows.

| output field name | explanation |
| --- | --- |
|global_n| number of nodes|
|global_m| number of edges|
|n_outliers| number of outliers (unclustered nodes)|
|node_coverage| proportion of outliers|
|mixing_parameter| mixing parameters (proportion of boundary edges) of nodes (each outlier has a mixing parameter of 1.0)|
|n| number of nodes internal to a cluster|
|m| number of edges internal to a cluster|
|c| number of edges on the boundary of a cluster (one end inside, one end outside)|
|conductance| conductances of clusters|
|degree_density| degree densities (\|E(S)\|/\|S\|) of clusters |
|edge_density| edge densities (\|E(S)\|/C(\|S\|, 2)) of clusters|
|mincut| sizes of minimum edge-cut of clusters|
|modularity| modularity scores of clusters|
|n_clusters| number of clusters|
|n_disconnected_clusters| number of disconnected clusters (with mincut = 0)|
|n_connected_clusters| number of connected clusters (with mincut > 0)|
|n_wellconnected_clusters| number of well-connected clusters (with mincut > log10(n))|

Additionally, there are `node.idx` and `com.idx` file that contain the original node and cluster ids. Each line in these files corresponds to the same line in the other output files. There is also a `outliers.txt` file that contains the node ids of the outliers.