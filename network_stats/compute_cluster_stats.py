import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymincut.pygraph import PyGraph

NODE_COLUMN_NAMES = [
    "node_id",
    "node",
    "vertex_id",
    "vertex",
    "source_id",
    "source",
    "target_id",
    "target",
    "node1_id",
    "node1",
    "node2_id",
    "node2",
]

COMMUNITY_COLUMN_NAMES = [
    "community_id",
    "community",
    "cluster_id",
    "cluster",
    "com_id",
    "com",
]


def compute_global_n(neighbors):
    """Compute the total number of nodes in the network."""
    return len(neighbors)


def compute_global_m(neighbors):
    """Compute the total number of edges in the network."""
    return sum(len(neighbors[node]) for node in neighbors) // 2


def compute_n_outliers(outliers):
    """Compute the number of outlier nodes."""
    return len(outliers)


def compute_node_coverage(outliers, global_n=None):
    """Compute the node coverage of the community."""
    global_n = compute_global_n(neighbors) if global_n is None else global_n
    n_outliers = len(outliers)
    return 1 - n_outliers / global_n if n_outliers > 0 else 1.0


def compute_mS_cS(neighbors, com):
    """
    Compute both the number of edges within the community (mS)
    and the number of edges on the boundary of the community (cS).
    Returns (mS, cS).
    """
    m_count = 0
    c_count = 0
    for node in com:
        for neighbor in neighbors.get(node, []):
            if neighbor in com:
                m_count += 1
            else:
                c_count += 1
    return m_count // 2, c_count


def compute_mS(neighbors, com):
    """Compute the number of edges within the community."""
    return compute_mS_cS(neighbors, com)[0]


def compute_cS(neighbors, com):
    """Compute the number of edges on the boundary of the community."""
    return compute_mS_cS(neighbors, com)[1]


def compute_nS(com):
    """Compute the number of nodes in the community."""
    return len(com)


def compute_conductance(neighbors, com, m=None, c=None):
    """Compute the conductance of the community."""
    m, c = compute_mS_cS(neighbors, com) if m is None or c is None else (m, c)
    return c / (2 * m + c) if c > 0 else 0.0


def compute_degree_density(neighbors, com, m=None, n=None):
    """Compute the degree density of the community."""
    m = compute_mS(neighbors, com) if m is None else m
    n = compute_nS(com) if n is None else n
    return m / n if m > 0 else 0.0


def compute_edge_density(neighbors, com, m=None, n=None):
    """Compute the edge density of the community."""
    m = compute_mS(neighbors, com) if m is None else m
    n = compute_nS(com) if n is None else n
    return 2 * m / (n * (n - 1)) if m > 0 else 0.0


def compute_mincut(neighbors, com):
    """Compute the mincut of the community."""
    cluster_edges = set()
    for node in com:
        for neighbor in neighbors.get(node, []):
            if neighbor in com:
                cluster_edges.add((node, neighbor))
    cluster_nodes = list(com)
    cluster_edges = list(cluster_edges)
    sub_G = PyGraph(cluster_nodes, cluster_edges)
    return sub_G.mincut("noi", "bqueue", False)[2]


def compute_modularity(neighbors, com, global_m=None, m=None, c=None):
    """Compute the modularity of the community."""
    m, c = compute_mS_cS(neighbors, com) if m is None or c is None else (m, c)
    global_m = compute_global_m(neighbors) if global_m is None else global_m
    return (m / global_m) - ((2 * m + c) / (2 * global_m)) ** 2


def compute_mixing_parameter(node, neighbors, node2coms, outliers):
    """Compute the mixing parameter for a node."""
    if node in outliers:
        return 1.0
    assert node in node2coms, f"Node {node} not found in node2coms mapping."
    coms = node2coms[node]
    n_neighbors = 0
    n_in = 0
    for neighbor in neighbors.get(node, set()):
        n_neighbors += 1
        n_in += 1 if coms.intersection(node2coms.get(neighbor, set())) else 0
    return 1.0 - n_in / n_neighbors if n_in > 0 else 0.0


def compute_n_clusters(com_iid_count):
    """Compute the number of clusters."""
    return com_iid_count


def compute_n_singleton_clusters(cluster_stats):
    """Compute the number of singleton clusters (cluster with n = 1)."""
    return sum(1 for n in cluster_stats["n"] if n == 1)


def compute_n_disconnected_clusters(cluster_stats):
    """Compute the number of disconnected clusters (cluster with n > 1 and mincut = 0)."""
    return sum(
        1
        for n, min_cut in zip(cluster_stats["n"], cluster_stats["mincut"])
        if n > 1 and min_cut == 0
    )


def compute_n_connected_clusters(cluster_stats):
    """Compute the number of connected clusters (cluster with n > 1 and mincut > 0)."""
    return sum(
        1
        for n, min_cut in zip(cluster_stats["n"], cluster_stats["mincut"])
        if n > 1 and min_cut > 0
    )


def compute_n_wellconnected_clusters(cluster_stats):
    """Compute the number of well-connected clusters (cluster with n > 1 and mincut > log10(n))."""
    return sum(
        1
        for n, min_cut in zip(cluster_stats["n"], cluster_stats["mincut"])
        if n > 1 and min_cut > np.log10(n)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cluster stats from network and community files."
    )
    parser.add_argument(
        "--network", type=str, required=True, help="Path to the network file"
    )
    parser.add_argument(
        "--community", type=str, required=True, help="Path to the community file"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory to save output files"
    )
    return parser.parse_args()


def detect_delimiter(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if "," in line:
                return ","
            if "\t" in line:
                return "\t"
            if " " in line:
                return " "
            break
    print("[WARNING] Could not detect delimiter...")
    return ","


args = parse_args()

network_fp = Path(args.network)
community_fp = Path(args.community)
output_dir = Path(args.outdir)

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Load community data
delimiter = detect_delimiter(community_fp)

node_id2iid = {}
node_iid2id = {}
node_iid_count = 0

com_id2iid = {}
com_iid2id = {}
com_iid_count = 0

node2coms = {}
com2nodes = {}

with open(community_fp, "r") as f:
    for line in f:
        # Skip comments
        if line.startswith("#"):
            continue

        # Skip empty lines
        if not line.strip():
            continue

        parts = line.strip().split(delimiter)

        assert (
            len(parts) == 2
        ), "Each line in the community file should contain two parts: node ID and community ID."

        if (
            parts[0].lower() in NODE_COLUMN_NAMES
            or parts[1].lower() in COMMUNITY_COLUMN_NAMES
        ):
            continue

        node_id = parts[0]
        if node_id not in node_id2iid:
            node_id2iid[node_id] = node_iid_count
            node_iid2id[node_iid_count] = node_id
            node_iid_count += 1
        node_iid = node_id2iid[node_id]

        com_id = parts[1]
        if com_id not in com_id2iid:
            com_id2iid[com_id] = com_iid_count
            com_iid2id[com_iid_count] = com_id
            com_iid_count += 1
        com_iid = com_id2iid[com_id]

        node2coms.setdefault(node_iid, set()).add(com_iid)
        com2nodes.setdefault(com_iid, set()).add(node_iid)

# Load network data
delimiter = detect_delimiter(network_fp)

neighbors = {}
outliers = set()

with open(network_fp, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split(delimiter)

        if (
            len(parts) >= 2
            and parts[0].lower() in NODE_COLUMN_NAMES
            and parts[1].lower() in NODE_COLUMN_NAMES
        ):
            continue

        assert len(parts) >= 2, (
            "Each line in the network file should contain at least two parts (node1 and node2). "
            + "There may be additional parts for edge weights or attributes."
        )

        node1, node2 = parts[0], parts[1]

        # Skip self-loops
        if node1 == node2:
            print("[WARNING] Skipping self-loop for node:", node1)
            continue

        # Skip duplicate edges
        node1_iid = node_id2iid.get(node1)
        node2_iid = node_id2iid.get(node2)
        if (
            node1_iid is not None
            and node2_iid is not None
            and node2_iid in neighbors.get(node1_iid, set())
        ):
            print(
                "[WARNING] Skipping duplicate edge between nodes: ",
                node1,
                " and ",
                node2,
            )
            continue

        if node1 not in node_id2iid:
            node_iid = node_iid_count
            outliers.add(node_iid)
            node_id2iid[node1] = node_iid
            node_iid2id[node_iid] = node1
            node_iid_count += 1

        if node2 not in node_id2iid:
            node_iid = node_iid_count
            outliers.add(node_iid)
            node_id2iid[node2] = node_iid
            node_iid2id[node_iid] = node2
            node_iid_count += 1

        node1_iid = node_id2iid[node1]
        node2_iid = node_id2iid[node2]

        neighbors.setdefault(node1_iid, set()).add(node2_iid)
        neighbors.setdefault(node2_iid, set()).add(node1_iid)

# Export node_id2iid to file (one column, no header, line id is the iid)
with (output_dir / "node.idx").open("w") as f:
    for iid in range(node_iid_count):
        f.write(f"{node_iid2id[iid]}\n")

# Export com_id2iid to file (one column, no header, line id is the iid)
with (output_dir / "com.idx").open("w") as f:
    for iid in range(com_iid_count):
        f.write(f"{com_iid2id[iid]}\n")

# Export outliers to file (one column, no header, line id is the iid)
with (output_dir / "outliers.txt").open("w") as f:
    for iid in sorted(outliers):
        f.write(f"{node_iid2id[iid]}\n")

# Compute global stats
global_stats = {
    "global_n": None,
    "global_m": None,
    "node_coverage": None,
}

global_n = compute_global_n(neighbors)
global_m = compute_global_m(neighbors)
n_outliers = compute_n_outliers(outliers)
node_coverage = compute_node_coverage(outliers, global_n)

global_stats["global_n"] = global_n
global_stats["global_m"] = global_m
global_stats["node_coverage"] = node_coverage

# Compute node stats and output a dataframe
node_stats = {
    "mixing_parameter": [],
}

for node_iid in range(node_iid_count):
    mixing_param = compute_mixing_parameter(node_iid, neighbors, node2coms, outliers)
    node_stats["mixing_parameter"].append(mixing_param)

# Compute cluster stats for each community and output a dataframe
cluster_stats = {
    "m": [],
    "n": [],
    "c": [],
    "conductance": [],
    "degree_density": [],
    "edge_density": [],
    "mincut": [],
    "modularity": [],
}

for com_iid in range(com_iid_count):
    assert (
        com_iid in com2nodes
    ), f"Community IID {com_iid} not found in com2nodes mapping."
    com = com2nodes[com_iid]

    n = compute_nS(com)
    m, c = compute_mS_cS(neighbors, com)
    cond = compute_conductance(neighbors, com, m, c)
    degree_density = compute_degree_density(neighbors, com, m, n)
    edge_density = compute_edge_density(neighbors, com, m, n)
    min_cut = compute_mincut(neighbors, com)
    modularity = compute_modularity(neighbors, com, global_m, m, c)

    cluster_stats["n"].append(n)
    cluster_stats["m"].append(m)
    cluster_stats["c"].append(c)
    cluster_stats["conductance"].append(cond)
    cluster_stats["degree_density"].append(degree_density)
    cluster_stats["edge_density"].append(edge_density)
    cluster_stats["mincut"].append(min_cut)
    cluster_stats["modularity"].append(modularity)

# Compute global stats
global_stats["n_clusters"] = compute_n_clusters(com_iid_count)
global_stats["n_singleton_clusters"] = compute_n_singleton_clusters(cluster_stats)
global_stats["n_disconnected_clusters"] = compute_n_disconnected_clusters(cluster_stats)
global_stats["n_connected_clusters"] = compute_n_connected_clusters(cluster_stats)
global_stats["n_wellconnected_clusters"] = compute_n_wellconnected_clusters(
    cluster_stats
)

# Export all stats to files
all_stats = {**global_stats, **node_stats, **cluster_stats}
for key, value in all_stats.items():
    pd.Series(value if isinstance(value, list) else [value]).to_csv(
        output_dir / f"{key}.txt", index=False, header=False
    )

# Create a done file to indicate completion
(output_dir / "done").touch()
