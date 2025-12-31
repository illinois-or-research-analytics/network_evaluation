REAL_NETWORK_PATH=$1
GENERATED_NETWORK_PATH=$2
CLUSTER_PATH=$3

# Get the file name without the path and extension
REAL_NETWORK_NAME=$(basename "$REAL_NETWORK_PATH" | sed 's/\.[^.]*$//')
GENERATED_NETWORK_NAME=$(basename "$GENERATED_NETWORK_PATH" | sed 's/\.[^.]*$//')

RUN_NAME=$4

OUTPUT=${REAL_NETWORK_NAME}_${GENERATED_NETWORK_NAME}_${RUN_NAME}_stats

if [ -z "$REAL_NETWORK_PATH" ] || [ -z "$GENERATED_NETWORK_PATH" ] || [ -z "$CLUSTER_PATH" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <real_network_path> <generated_network_path> <cluster_path> <output>"
    exit 1
fi

echo "Real Network Path: $REAL_NETWORK_PATH"
echo "Generated Network Path: $GENERATED_NETWORK_PATH"
echo "Cluster Path: $CLUSTER_PATH"
echo "Output Directory: $OUTPUT"

python3 ./network_stats/compute_stats.py --input-network "$REAL_NETWORK_PATH" \
                                         --input-clustering "$CLUSTER_PATH" \
                                         --output-folder "orig_$OUTPUT" \
                                         --overwrite &

python3 ./network_stats/compute_stats.py --input-network "$GENERATED_NETWORK_PATH" \
                                         --input-clustering "$CLUSTER_PATH" \
                                         --output-folder "syn_$OUTPUT" \
                                         --overwrite &
wait

python3 ./compare_networks/compare_stats_pair.py --network-1-folder "orig_$OUTPUT" \
                                                 --network-2-folder "syn_$OUTPUT" \
                                                 --output-file "${OUTPUT}.csv" \
                                                 --is-compare-sequence
