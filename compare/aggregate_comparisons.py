# ==============================================================================
# Network Statistics Aggregator (aggregate_stats.py)
# ==============================================================================
# This script searches through a structured directory of synthetic network
# outputs, aggregates individual comparison CSV files using multiprocessing,
# and compiles them into a single master CSV. It also generates a completion
# summary identifying missing network replicates.
#
# EXPECTED FILE STRUCTURE:
#   <root> / <generator> / <clustering_id> / <network_id> / <run_id> / <comp_fn>
#
# USAGE:
#   python aggregate_stats.py [OPTIONS] --root <dir> \
#                             --network-fp <file> --generators <gens...> --clusterings <clusters...>
#                             --output <file>
#
# REQUIRED ARGUMENTS:
#   --root <dir>             : Root directory for the stats.
#   --network-fp <file>      : File containing the list of network IDs to process.
#   --generators <gens...>   : Space-separated list of generator IDs (e.g., ec-sbm, lfr).
#   --clusterings <ids...>   : Space-separated list of clustering IDs to gather.
#   --output <file>          : Output path for the aggregated data.
#
# OPTIONAL ARGUMENTS:
#   --comp-fns <files...>    : Target CSV filenames to aggregate (default: comparison.csv).
#   --num-replicates <int>   : Number of run/replicate IDs to collect per network (default: 1).
#   --n-procs <int>          : Number of CPU processes for data aggregation (default: 16).
#
# EXAMPLES:
#   python aggregate_stats.py \
#       --root data/synthetic_networks/stats/ \
#       --output plots/agg_comp.csv \
#       --network-fp data/networks.txt \
#       --generators ec-sbm lfr \
#       --clusterings leiden-cpm-0.5 sbm-flat-best
# ==============================================================================

import argparse
import logging
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Constants ---

MAPPING = {
    "ec-sbm-v2": "EC-SBMv2",
    "ec-sbm-v1": "EC-SBMv1",
    "ec-sbm": "EC-SBM",
    "abcd+o": "ABCD+o",
    "sbm": "SBM",
    "lfr": "LFR",
}

# --- Helper Functions ---


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate network statistics comparison CSV files."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for the synthetic network stats.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the aggregated data.",
    )
    parser.add_argument(
        "--network-fp",
        type=str,
        required=True,
        help="File containing the list of network IDs to process.",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        required=True,
        help="List of generator IDs (e.g., ec-sbm-v2, lfr).",
    )
    parser.add_argument(
        "--clusterings",
        nargs="+",
        required=True,
        help="List of clustering IDs to gather (e.g., leiden-cpm-0.5, leiden-mod+cm).",
    )
    parser.add_argument(
        "--comp-fns",
        nargs="+",
        default=["comparison.csv"],
        help="Comparison filenames to look for inside each replicate folder.",
    )
    parser.add_argument(
        "--num-replicates",
        type=int,
        default=1,
        help="Number of run/replicate IDs to collect per network.",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=16,
        help="Number of processes to use for data aggregation.",
    )
    return parser.parse_args()


def _process_task(args: tuple) -> Dict[str, Any]:
    """
    Worker function to process a single network_id, generator, and clustering_id combination.
    """
    (
        network_id,
        generator,
        clustering_id,
        run_id,
        root,
        comp_fns,
    ) = args

    # Expected Structure:
    # root / generator / clustering_id / network_id / run_id / comp_fn
    path_components = [
        generator,
        clustering_id,
        network_id,
        str(run_id),
    ]

    target_dir = root.joinpath(*path_components)

    df = None
    success = False

    if target_dir.exists():
        for comp_fn in comp_fns:
            comp_path = target_dir / comp_fn
            if comp_path.exists():
                try:
                    df_tmp = pd.read_csv(comp_path)
                    if not df_tmp.empty:
                        success = True
                        if df is None:
                            df = df_tmp
                        else:
                            df = pd.concat([df, df_tmp], ignore_index=True)
                except Exception as e:
                    logger.warning(f"Error reading {comp_path}: {e}")

    mapped_generator = MAPPING.get(generator, generator)

    if success and df is not None:
        df["network_id"] = network_id
        df["generator_id"] = generator
        df["simulator"] = mapped_generator
        df["clustering_id"] = clustering_id
        df["run_id"] = run_id

    return {
        "success": success,
        "network_id": network_id,
        "generator": mapped_generator,
        "clustering_id": clustering_id,
        "run_id": run_id,
        "data": df,
    }


def collect_dataframe(
    network_ids: List[str],
    generators: List[str],
    clustering_ids: List[str],
    num_replicates: int,
    root: Path,
    comp_fns: List[str],
    max_workers: int = 16,
) -> pd.DataFrame:

    tasks = []
    for network_id in network_ids:
        for generator in generators:
            for clustering_id in clustering_ids:
                for run_id in range(num_replicates):
                    tasks.append(
                        (
                            network_id,
                            generator,
                            clustering_id,
                            run_id,
                            root,
                            comp_fns,
                        )
                    )

    logger.info(
        f"Starting data collection for {len(tasks)} tasks using {max_workers} workers."
    )

    df_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_process_task, task): task for task in tasks}

        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Aggregating CSVs",
            unit="file",
        ):
            try:
                result = future.result()
                if result["success"] and result["data"] is not None:
                    df_data.append(result["data"])
            except Exception as e:
                task_info = future_to_task[future]
                logger.error(
                    f"Task failed for Net: {task_info[0]}, Gen: {task_info[1]}. Error: {e}"
                )

    if not df_data:
        logger.error("No data collected. Please check paths and inputs.")
        return pd.DataFrame()

    logger.info(f"Collected {len(df_data)} valid dataframes. Concatenating...")
    df = pd.concat(df_data, ignore_index=True)

    # ProcessPoolExecutor.as_completed yields in completion order, so df_data
    # was appended in non-deterministic order — sort by task identity so the
    # output CSV is byte-stable across runs.  kind='stable' preserves each
    # per-task row block's internal order.
    sort_cols = [c for c in ("network_id", "generator_id", "clustering_id", "run_id") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return df


def print_completion_summary(
    df: pd.DataFrame,
    all_networks: List[str],
    generators: List[str],
    clustering_ids: List[str],
):
    """
    Groups completion by Generator -> Clustering ID and lists missing networks.
    """
    logger.info("Computing completion statistics...")
    total_expected = len(all_networks)
    expected_set = set(all_networks)

    print("\n" + "=" * 80)
    print(f"COMPLETION SUMMARY (Total Expected Networks per Pair: {total_expected})")
    print("=" * 80)

    # Convert mapping generators for lookup if dataframe uses mapped names
    mapped_generators = [MAPPING.get(g, g) for g in generators]

    for generator in mapped_generators:
        print(f"\nGenerator: {generator}")
        print("-" * 80)

        for clustering_id in clustering_ids:
            if df.empty:
                finished_set = set()
            else:
                # A network is 'finished' for this gen/cluster pair if it appears in the dataframe
                subset = df[
                    (df["simulator"] == generator)
                    & (df["clustering_id"] == clustering_id)
                ]
                finished_set = set(subset["network_id"].unique())

            missing_set = expected_set - finished_set
            missing_list = sorted(list(missing_set))

            count_str = f"{len(finished_set)}/{total_expected}"
            print(f"  Clustering: {clustering_id:<25} | Count: {count_str}")

            if missing_list:
                prefix = "    Missing: "
                wrapper = textwrap.TextWrapper(
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix),
                    width=80,
                )
                print(wrapper.fill(", ".join(missing_list)))

    print("\n" + "=" * 80 + "\n")


# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()

    root = Path(args.root)

    network_fp = Path(args.network_fp)
    if not network_fp.exists():
        logger.critical(f"Network list file not found: {network_fp}")
        sys.exit(1)

    with open(network_fp, "r") as f:
        network_ids = [line.strip() for line in f.readlines() if line.strip()]

    df = collect_dataframe(
        network_ids=network_ids,
        generators=args.generators,
        clustering_ids=args.clusterings,
        num_replicates=args.num_replicates,
        root=root,
        comp_fns=args.comp_fns,
        max_workers=args.n_procs,
    )

    if not df.empty:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

    # Print Global Completion Summary even if dataframe is empty (will show 0/N for all)
    print_completion_summary(
        df=df,
        all_networks=network_ids,
        generators=args.generators,
        clustering_ids=args.clusterings,
    )

    logger.info("Processing complete.")
