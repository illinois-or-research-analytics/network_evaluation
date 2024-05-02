import os
from typing import Any, Dict

import pandas as pd
import matplotlib.pyplot as plt


def plot_distr_scatter(
        df: pd.DataFrame,
        colname: str,
        ax: plt.Axes,
        **plot_kwargs: Dict[str, Any],
) -> None:
    # Compute the frequency
    df_plot = df.groupby(colname).size().reset_index(name='count')

    # Plot the distribution
    sns.scatterplot(
        ax=ax,
        data=df_plot,
        x=colname,
        y='count',
        **plot_kwargs,
    )


def compute_distr_stats(
    df: pd.DataFrame,
    colname: str,
) -> Dict[str, float]:
    df_stats = df[colname]

    mean = df_stats.mean()
    std = df_stats.std()

    min_ = df_stats.min()
    q1 = df_stats.quantile(0.25)
    med = df_stats.median()
    q3 = df_stats.quantile(0.75)
    max_ = df_stats.max()

    return {
        'mean': mean,
        'std': std,
        'min': min_,
        'q1': q1,
        'med': med,
        'q3': q3,
        'max': max_,
    }


def parse_mapping(
    mapping_fn: str,
) -> Dict[str, int]:
    mapping = dict()

    if not os.path.exists(mapping_fn):
        return mapping

    f = open(mapping_fn)
    csv_reader = csv.reader(f, delimiter='\t')

    for i, u in enumerate(csv_reader):
        mapping[u] = i

    f.close()

    return mapping
