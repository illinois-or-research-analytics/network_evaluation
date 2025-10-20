import argparse
from pathlib import Path

import pandas as pd
from pymincut.pygraph import PyGraph

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

args = parser.parse_args()

network_fp = Path(args.network)
community_fp = Path(args.community)
output_dir = Path(args.outdir)

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)


# Detect delimiters
def detect_delimiter(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline()
        if "," in first_line:
            return ","
        elif "\t" in first_line:
            return "\t"
        elif " " in first_line:
            return " "
        else:
            raise ValueError("No delimiter found in the first line of the file.")


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
    lines = f.readlines()
    for line in lines:
        if line.startswith("#"):
            continue

        parts = line.strip().split(delimiter)
        assert (
            len(parts) == 2
        ), "Each line in the community file should contain exactly two parts: node ID and community ID."

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
    lines = f.readlines()
    for line in lines:
        if line.startswith("#"):
            continue

        parts = line.strip().split(delimiter)
        assert (
            len(parts) == 2
        ), "Each line in the network file should contain exactly two nodes."

        node1, node2 = parts[0], parts[1]

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
    for iid in outliers:
        f.write(f"{node_iid2id[iid]}\n")


def compute_global_n(neighbors):
    """Compute the total number of nodes in the network."""
    return len(neighbors)


def compute_global_m(neighbors):
    """Compute the total number of edges in the network."""
    return sum(len(neighbors[node]) for node in neighbors) // 2


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
    return compute_mS_cS(neighbors, com)[0]


def compute_cS(neighbors, com):
    return compute_mS_cS(neighbors, com)[1]


def compute_nS(com):
    """Compute the number of nodes in the community."""
    return len(com)


def compute_conductance(neighbors, com, m=None, c=None):
    """Compute the conductance of the community."""
    m, c = compute_mS_cS(neighbors, com) if m is None or c is None else (m, c)
    return c / (2 * m + c) if c > 0 else 0.0


def compute_normalized_density(neighbors, com, m=None, n=None):
    """Compute the normalized density of the community."""
    m = compute_mS(neighbors, com) if m is None else m
    n = compute_nS(neighbors, com) if n is None else n
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


# Compute global stats
global_stats = {
    "global_n": None,
    "global_m": None,
    "node_coverage": None,
}

global_n = compute_global_n(neighbors)
global_m = compute_global_m(neighbors)
node_coverage = compute_node_coverage(outliers, global_n)

global_stats["global_n"] = global_n
global_stats["global_m"] = global_m
global_stats["node_coverage"] = node_coverage

for key in global_stats:
    pd.Series([global_stats[key]]).to_csv(
        output_dir / f"{key}.txt", index=False, header=False
    )

# Compute node stats and output a dataframe
node_stats = {
    "mixing_parameter": [],
}

for node_iid in range(node_iid_count):
    mixing_param = compute_mixing_parameter(node_iid, neighbors, node2coms, outliers)

    node_stats["mixing_parameter"].append(mixing_param)

for key in node_stats:
    pd.Series(node_stats[key]).to_csv(
        output_dir / f"{key}.txt", index=False, header=False
    )

# Compute cluster stats for each community and output a dataframe
cluster_stats = {
    "m": [],
    "n": [],
    "c": [],
    "conductance": [],
    "normalized_density": [],
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
    norm_density = compute_normalized_density(neighbors, com, m, n)
    min_cut = compute_mincut(neighbors, com)
    modularity = compute_modularity(neighbors, com, global_m, m, c)

    cluster_stats["n"].append(n)
    cluster_stats["m"].append(m)
    cluster_stats["c"].append(c)
    cluster_stats["conductance"].append(cond)
    cluster_stats["normalized_density"].append(norm_density)
    cluster_stats["mincut"].append(min_cut)
    cluster_stats["modularity"].append(modularity)

for key in cluster_stats:
    pd.Series(cluster_stats[key]).to_csv(
        output_dir / f"{key}.txt", index=False, header=False
    )

# Create a done file to indicate completion
(output_dir / "done").touch()
