# ==============================================================================
# Network Statistics Visualizer (visualize_stats.py)
# ==============================================================================
# This script reads an aggregated comparison CSV, filters the data based on
# specified generator/clustering methods, and computes summary statistics
# across run replicates. It outputs a comprehensive summary CSV and a PDF
# grid of boxplots for the requested network statistics.
#
# USAGE:
#   python visualize_stats.py [OPTIONS] --data-fp <file> --network-fp <file> \
#                             --output-dir <dir> --output-fn <name>
#
# REQUIRED ARGUMENTS (Group 1: Methods - Must be equal length):
#   --generators <gens...>   : List of generator IDs (e.g., ec-sbm).
#   --clusterings <ids...>   : List of clustering IDs (e.g., leiden-cpm).
#   --names <names...>       : Display names for the (generator, clustering) pairs.
#
# REQUIRED ARGUMENTS (Group 2: Metrics - Must be equal length):
#   --stats <stats...>       : List of stat IDs (e.g., mean_degree, modularity).
#   --types <types...>       : List of stat types (e.g., scalar, distribution).
#   --metrics <metrics...>   : List of distance metrics (e.g., abs_diff, emd).
#
# REQUIRED ARGUMENTS (Group 3: Data):
#   --data-fp <file>         : Path to the aggregated comparison CSV.
#   --network-fp <file>      : File containing the whitelist of network IDs.
#   --output-dir <dir>       : Directory to save outputs.
#   --output-fn <name>       : Base filename for outputs.
#
# OPTIONAL ARGUMENTS:
#   --hide-fliers            : Flag to hide outliers in the generated boxplots.
#
# EXAMPLES:
#   1. Visualizing Two Methods across Scalar and Distribution Stats:
#      python visualize_stats.py --data-fp results/agg_comp.csv \
#        --network-fp data/networks.txt \
#        --generators ec-sbm abcd+o --clusterings leiden-cpm-0.5 sbm-flat-best+wcc \
#        --names "EC-SBM x Leiden-CPM(0.5)" "ABCD+o x SBM-Flat+WCC" \
#        --stats mean_degree degree --types scalar distribution \
#        --metrics rel_diff emd --output-dir final_plots --output-fn compare --hide-fliers
# ==============================================================================

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Constants ---

STAT_NAME_MAPPING = {
    # Clustering Scalar Stats
    "node_coverage": "Node Coverage",
    "n_outliers": "Number of Outliers",
    "n_clusters": "Number of Clusters",
    "n_disconnected_clusters": "Number of Disconnected Clusters",
    "n_connected_clusters": "Number of Connected Clusters",
    "n_wellconnected_clusters": "Number of Well-Connected Clusters",
    # Clustering Node Distribution Stats
    "mixing_parameter": "Mixing Parameter",
    # Clustering Cluster Distribution Stats
    "m": "Number of Internal Edges",
    "n": "Number of Nodes in Cluster",
    "c": "Number of Boundary Edges",
    "conductance": "Conductance",
    "degree_density": "Degree Density",
    "edge_density": "Edge Density",
    "mincut": "Min Cut Sizes",
    "modularity": "Modularity",
    # Network Scalar Stats
    "n_nodes": "Number of Nodes",
    "n_edges": "Number of Edges",
    "n_concomp": "Number of Connected Components",
    "mean_degree": "Mean Degree",
    "deg_assort": "Degree Assortativity",
    "mean_kcore": "Mean K-Core",
    "global_ccoeff": "Global Clustering Coeff",
    "local_ccoeff": "Avg Local Clustering Coeff",
    "pseudo_diameter": "Pseudo-Diameter",
    "char_time": "Characteristic Time",
    "node_percolation_targeted": "Node Percolation (Targeted)",
    "node_percolation_random": "Node Percolation (Random)",
    "frac_giant_ccomp": "Fraction of Giant Connected Component",
    # Network Distribution Stats
    "degree": "Degree of Nodes",
    "local_ccoeff_nodes": "Clustering Coeffs of Nodes",
    "pagerank": "PageRank of Nodes",
    "betweenness": "Betweenness Centrality of Nodes",
    "kcore": "K-Core of Nodes",
    "concomp_sizes": "Sizes of Connected Components",
}

# Map specific (stat_type, distance_type) combinations to acronyms
DISTANCE_MAPPING = {
    ("scalar", "rel_diff"): "SRD",
    ("scalar", "abs_diff"): "SAD",
    ("sequence", "rmse"): "RMSE",
    ("sequence", "mean_l1"): "MAE",
    ("distribution", "emd"): "EMD",
    ("distribution", "ks"): "KS",
}

# --- Helper Functions ---


def q1(x):
    """Calculate the 25th percentile."""
    return x.quantile(0.25)


def q3(x):
    """Calculate the 75th percentile."""
    return x.quantile(0.75)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize compared network statistics from aggregated data."
    )
    parser.add_argument(
        "--data-fp",
        type=str,
        required=True,
        help="Path to the aggregated comparison CSV.",
    )
    parser.add_argument(
        "--generators", nargs="+", required=True, help="List of generator IDs."
    )
    parser.add_argument(
        "--clusterings", nargs="+", required=True, help="List of clustering IDs."
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="List of display names for the (generator, clustering) pairs.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        required=True,
        help="List of stat IDs (e.g., deg_assort, mincuts).",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        required=True,
        help="List of stat types (e.g., scalar, sequence, distribution).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="List of distance metrics (e.g., abs_diff, rmse, emd).",
    )
    parser.add_argument(
        "--network-fp",
        type=str,
        required=True,
        help="File containing whitelist of network IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        default="plots",
        help="Directory to save output plots and summaries.",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        required=True,
        default="comparison_results",
        help="Base filename for outputs (without extension).",
    )
    parser.add_argument(
        "--hide-fliers", action="store_true", help="Hide outliers in the boxplots."
    )
    return parser.parse_args()


def get_common_networks_df(
    df_wide: pd.DataFrame,
    methods: List[str],
    metric_col: str,
    network_list: List[str],
) -> pd.DataFrame:
    """Filters the DataFrame for a specific metric to strictly include valid intersections."""
    df_filtered = df_wide[
        (df_wide["network_id"].isin(network_list)) & (df_wide["Method"].isin(methods))
    ].copy()

    # Drop rows where the metric value is NaN
    df_metric = df_filtered[["network_id", "Method", metric_col]].dropna()

    # Count how many valid methods exist for each network
    method_counts = df_metric.groupby("network_id", observed=True)["Method"].nunique()
    valid_networks = method_counts[method_counts == len(methods)].index

    # Print detailed reasons for filtered networks
    filtered_out_networks = set(network_list) - set(valid_networks)
    if filtered_out_networks:
        logger.info(
            f"[{metric_col}] Filtered out {len(filtered_out_networks)} network(s): {sorted(list(filtered_out_networks))}"
        )

        # for net in sorted(list(filtered_out_networks)):
        #     # Find which methods actually have valid data for this network
        #     present_methods = df_metric[df_metric["network_id"] == net][
        #         "Method"
        #     ].unique()
        #     missing_methods = set(methods) - set(present_methods)

        #     logger.info(
        #         f"  -> '{net}' is missing valid data for: {sorted(list(missing_methods))}"
        #     )
    else:
        logger.info(
            f"[{metric_col}] All {len(network_list)} networks retained (no filtering required)."
        )

    df_clean = df_metric[df_metric["network_id"].isin(valid_networks)].copy()

    df_clean["Method"] = pd.Categorical(
        df_clean["Method"], categories=methods, ordered=True
    )
    df_clean = df_clean.sort_values(["network_id", "Method"]).reset_index(drop=True)

    return df_clean


def generate_summary_table(
    df_wide: pd.DataFrame,
    methods: List[str],
    metrics: List[str],
    metric_names: List[str],
    network_whitelist: List[str],
    output_dir: Path,
    output_fn: str,
):
    """Generates a summary CSV containing standard statistics and 'count_best'."""
    logger.info(f"Generating Summary Table: summary_{output_fn}.csv")

    series_list = []

    for metric, metric_name in zip(metrics, metric_names):
        df_clean = get_common_networks_df(df_wide, methods, metric, network_whitelist)

        if df_clean.empty:
            logger.warning(f"No valid intersection found for {metric_name}. Skipping.")
            continue

        stats = df_clean.groupby("Method", observed=True)[metric].agg(
            ["count", "min", q1, "median", q3, "max", "mean", "std"]
        )

        pivoted = df_clean.pivot(index="network_id", columns="Method", values=metric)
        min_vals = pivoted.min(axis=1)
        is_best = pivoted.eq(min_vals, axis=0)

        stats["count_best"] = is_best.sum(axis=0)

        stats = stats[
            ["count", "count_best", "min", "q1", "median", "q3", "max", "mean", "std"]
        ]

        stacked = stats.stack()
        stacked.name = metric_name

        series_list.append(stacked)

    if not series_list:
        logger.warning("No valid data generated for the summary table.")
        return

    final_summary = pd.concat(series_list, axis=1)
    final_summary.index.names = ["Method", "Statistic"]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"summary_{output_fn}.csv"

    final_summary.reset_index().to_csv(out_path, index=False)
    logger.info(f"Saved summary to {out_path}")


def plot_boxplots(
    df_wide: pd.DataFrame,
    methods: List[str],
    metrics: List[str],
    metric_names: List[str],
    network_whitelist: List[str],
    output_dir: Path,
    output_fn: str,
    hide_fliers: bool = False,
):
    logger.info(f"Generating Boxplots: {output_fn}.pdf")
    num_metrics = len(metrics)
    if num_metrics == 0:
        return

    # Dynamic row/column calculation for aesthetics
    if num_metrics <= 3:
        ncols, nrows = num_metrics, 1
    elif num_metrics == 4:
        ncols, nrows = 2, 2
    elif num_metrics <= 6:
        ncols, nrows = 3, 2
    elif num_metrics <= 8:
        ncols, nrows = 4, 2
    else:
        ncols = 4
        nrows = math.ceil(num_metrics / ncols)

    ncols_legend = 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        dpi=300,
    )

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    handles, labels = None, None

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        df_clean = get_common_networks_df(df_wide, methods, metric, network_whitelist)

        # Set x to the Stat name to cluster boxes tightly
        df_clean["Stat"] = metric_name

        ax.grid(True, which="major", axis="y", linestyle="--", color="lightgray")

        if not df_clean.empty:
            sns.boxplot(
                data=df_clean,
                x="Stat",
                y=metric,
                hue="Method",
                ax=ax,
                hue_order=methods,
                showmeans=False,
                showfliers=not hide_fliers,
                orient="v",
                boxprops={"edgecolor": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 1.5},
                capprops={"color": "black", "linewidth": 1.5},
                medianprops={"color": "red", "linewidth": 1.5},
                meanprops={
                    "marker": "^",
                    "markerfacecolor": "green",
                    "markeredgecolor": "green",
                    "markersize": 5,
                },
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "black",
                    "markeredgecolor": "black",
                    "markersize": 3,
                },
            )

            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

        ax.set_ylabel("Distance", fontsize=12)
        ax.set_xlabel("", fontsize=12)

        # Increase x-tick label size since it now shows the Stat name
        ax.tick_params(axis="x", labelsize=14, bottom=False)
        ax.axhline(y=0.0, color="blue", linestyle="--", linewidth=1, alpha=0.8)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for idx in range(num_metrics, len(axes)):
        fig.delaxes(axes[idx])

    # Reserve the top 12% of the figure exclusively for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.89),  # Sit precisely above the 88% cutoff line
            ncol=ncols_legend,
            fontsize=12,
            frameon=False,
            title="Methods",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_fn}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    logger.info(f"Saved plot to {out_path}")
    plt.close(fig)


# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()

    # 1. Validation
    if not (len(args.generators) == len(args.clusterings) == len(args.names)):
        logger.critical(
            "Arguments 'generators', 'clusterings', and 'names' must have the same length."
        )
        sys.exit(1)

    if not (len(args.stats) == len(args.types) == len(args.metrics)):
        logger.critical(
            "Arguments 'stats', 'types', and 'metrics' must have the same length."
        )
        sys.exit(1)

    # 2. Data Loading & Mapping Setup
    logger.info("Loading network valid list and aggregated data...")
    network_list_path = Path(args.network_fp)
    if not network_list_path.exists():
        logger.critical(f"Network list file not found: {network_list_path}")
        sys.exit(1)

    with open(network_list_path, "r") as f:
        network_whitelist = [line.strip() for line in f.readlines() if line.strip()]

    df_raw = pd.read_csv(args.data_fp)

    map_df = pd.DataFrame(
        {
            "generator_id": args.generators,
            "clustering_id": args.clusterings,
            "Method": args.names,
        }
    )
    df_mapped = df_raw.merge(map_df, on=["generator_id", "clustering_id"], how="inner")

    if df_mapped.empty:
        logger.warning(
            "No data found matching the provided generators and clusterings."
        )
        sys.exit(0)

    # 3. Aggregate across Replicates (Run IDs)
    logger.info("Averaging data across run replicates...")
    df_avg = df_mapped.groupby(
        ["network_id", "Method", "stat", "stat_type", "distance_type"], as_index=False
    )["distance"].mean()

    # 4. Pivot to Wide DataFrame using the requested Stats, Types, and Metrics
    logger.info("Pivoting dataframe for requested metrics...")
    df_wide = pd.DataFrame(columns=["network_id", "Method"])
    metric_cols = []
    metric_names = []

    for st, sty, smet in zip(args.stats, args.types, args.metrics):
        col_id = f"{st}_{sty}_{smet}"
        base_name = STAT_NAME_MAPPING.get(st, st)

        dist_abbr = DISTANCE_MAPPING.get((sty, smet))
        if not dist_abbr:
            if smet.lower() in ["emd", "rmse", "mae", "ks", "js"]:
                dist_abbr = smet.upper()
            else:
                dist_abbr = smet.replace("_", " ").title()

        display_name = f"{base_name} ({dist_abbr})"

        metric_cols.append(col_id)
        metric_names.append(display_name)

        slice_df = df_avg[
            (df_avg["stat"] == st)
            & (df_avg["stat_type"] == sty)
            & (df_avg["distance_type"] == smet)
        ].copy()

        slice_df = slice_df[["network_id", "Method", "distance"]].rename(
            columns={"distance": col_id}
        )

        if df_wide.empty:
            df_wide = slice_df
        else:
            df_wide = pd.merge(
                df_wide, slice_df, on=["network_id", "Method"], how="outer"
            )

    active_methods = df_wide.dropna(subset=metric_cols, how="all")["Method"].unique()
    valid_names = [m for m in args.names if m in active_methods]

    if len(valid_names) < len(args.names):
        dropped = set(args.names) - set(valid_names)
        logger.warning(f"Dropping methods with no completed networks: {dropped}")
        args.names = (
            valid_names  # Update global list to adjust strict intersection target
        )

    if not args.names:
        logger.error("No methods have completed networks. Exiting.")
        sys.exit(0)

    df_wide["Method"] = pd.Categorical(
        df_wide["Method"], categories=args.names, ordered=True
    )

    # 5. Output Generation
    output_dir = Path(args.output_dir)

    generate_summary_table(
        df_wide=df_wide,
        methods=args.names,
        metrics=metric_cols,
        metric_names=metric_names,
        network_whitelist=network_whitelist,
        output_dir=output_dir,
        output_fn=args.output_fn,
    )

    plot_boxplots(
        df_wide=df_wide,
        methods=args.names,
        metrics=metric_cols,
        metric_names=metric_names,
        network_whitelist=network_whitelist,
        output_dir=output_dir,
        output_fn=args.output_fn,
        hide_fliers=args.hide_fliers,  # Pass the new argument here
    )

    logger.info("Processing complete.")
