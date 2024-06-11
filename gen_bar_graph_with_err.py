import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESOLUTIONS = {'.001'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

# ex:
# root = 'output/abcdmcs_abcd/tables/'
# output = 'output/abcdmcs_abcd/plots/'


args = parse_args()
root = Path(args.root)
output = Path(args.output)

stats = {
    ('mincuts', 'distribution', 'ks'),
    ('diameter', 'scalar', 'rpd'),
    ('mixing_mus', 'distribution', 'ks'),
    ('global_ccoeff', 'scalar', 'rpd'),
    ('deg_assort', 'scalar', 'rpd'),
    ('degree', 'distribution', 'ks'),
    ('mixing_xi', 'scalar', 'rpd'),
    ('c_size', 'distribution', 'ks'),
    ('c_edges', 'distribution', 'ks'),
}

# Expected input:
# root
# ├── {network}_{resolution}.csv

# Output: output/{stat}_{stat_type}_{distance_type}.png

if not output.exists():
    output.mkdir(parents=True)
else:
    for fp in output.iterdir():
        fp.unlink()

means = dict()
errs = dict()

all_network_resolutions = set()
all_simulators = set()

for stat, stat_type, distance_type in stats:
    means.setdefault((stat, stat_type, distance_type), dict())
    errs.setdefault((stat, stat_type, distance_type), dict())

    mean = means[(stat, stat_type, distance_type)]
    err = errs[(stat, stat_type, distance_type)]

    for fp in root.iterdir():
        if fp.name.endswith('.csv'):
            network_resolution = fp.stem
            resolution = network_resolution.split('_')[-1]
            if resolution not in RESOLUTIONS:
                continue

            network = network_resolution[:-len(resolution) - 1]
            network_resolution = f'{network} 0{resolution}'
            all_network_resolutions.add(network_resolution)

            df = pd.read_csv(fp)
            df = df.loc[
                (df['stat'] == stat)
                & (df['stat_type'] == stat_type)
                & (df['distance_type'] == distance_type)
            ]

            for x in df.columns[3::3]:
                simulator = x.split('_')[-1]
                all_simulators.add(simulator)

                mean.setdefault(simulator, dict())
                mean[simulator].setdefault(network_resolution, dict())

                mean[simulator][network_resolution] = \
                    np.nan \
                    if df.empty \
                    else df[f'distance_mean_{simulator}'].values[0]

                err.setdefault(simulator, dict())
                err[simulator].setdefault(network_resolution, dict())
                err[simulator][network_resolution] = \
                    np.nan \
                    if df.empty \
                    else df[f'distance_std_{simulator}'].values[0]

    # Plot the results
all_network_resolutions = sorted(all_network_resolutions)
all_simulators = sorted(all_simulators)
n_simulators = len(all_simulators)

for stat, stat_type, distance_type in stats:
    mean = means[(stat, stat_type, distance_type)]
    err = errs[(stat, stat_type, distance_type)]

    labels = [
        simulator
        for simulator in all_simulators
    ]
    xticklabels = [
        network_resolution
        for network_resolution in all_network_resolutions
    ]
    xs = np.arange(len(xticklabels))
    ys = [
        [
            mean[simulator][network_resolution]
            for network_resolution in all_network_resolutions
        ]
        for simulator in all_simulators
    ]
    width = 0.4

    figsize = (10, len(xticklabels))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    M = -1
    for i, y in enumerate(ys):
        t = ax.barh(
            xs + (2 * i - 1) * width / 2, y,
            xerr=[
                err[all_simulators[i]][network_resolution]
                for network_resolution in all_network_resolutions
            ],
            height=width,
            label=labels[i],
        )
        ax.bar_label(t, padding=3, fmt='%.2f')

    ax.set_yticks(xs)
    ax.set_yticklabels(xticklabels)
    ax.set_ylim(- 0.5, len(xticklabels) - 1 + 0.5)
    if distance_type == 'ks':
        ax.set_xlim(0.0, 1.0 + 0.1)
    elif distance_type == 'rpd':
        ax.set_xlim(-1.0 - 0.2, 1.0 + 0.2)
    # else:
        # ax.set_xlim(0.0)
    ax.legend(fontsize='xx-large')
    ax.set_title(f'{distance_type} of {stat} ({stat_type})')

    fig.tight_layout()
    plt.savefig(output / f'{stat}_{stat_type}_{distance_type}.png')
