from soccer_network.data import (match_ids, huskies_player_ids, huskies_events, huskies_passes, matches_df, all_events,
                                 events_df)
from soccer_network.graphs import load_graphml
from graph_tool import clustering, Graph
import pandas as pd
import numpy as np
from typing import Tuple, List, Callable, Any

"""
clustering
motifs
centrality
flow
dynamics
configuration model
topology
"""


def metrics_passing_volume(g: Graph):
    vs = g.get_vertices()
    total = g.get_in_degrees(vs) + g.get_out_degrees(vs)
    return np.mean(total), np.std(total)


def post_passing_volume(results: Tuple):
    mean, sigma = zip(
        *results)  # https://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
    df = pd.DataFrame(dict(MatchID=match_ids, pv_mean=mean, pv_sigma=sigma))
    df = matches_df[['Outcome', 'OwnScore']].join(df, how='right')
    print('=' * 25)
    print('(Zoned) passing volume')
    print('=' * 25)
    print(df.corr())


def metric_motifs(g: Graph):
    return clustering.motif_significance(g, 3)


def post_motifs(results: Tuple):
    z_scores = results[1]
    print(z_scores)


# TODO: add your metrics here, the function should have only one required argument and can return anything you like
metrics = [
    metrics_passing_volume,
    metric_motifs
]

# TODO| add your post-metrics-computation processing function here,
# the results of every metric **for all graphs** are passed in as **one** single argument
# NOTE: the order of results is the same as the order of `match_ids`
post_metrics = [post_passing_volume]


def run_metric(gs: List[Graph], metric: Callable[[Graph], Any], post_metric: Callable):
    results = [metric(g) for g in gs]
    post_metric(results)


if __name__ == "__main__":
    graphs = [load_graphml('../graphs/network-{0}.xml'.format(mi)) for mi in match_ids]
    # graphs = [load_graphml('../graphs/zoned-network-{0}.xml'.format(mi)) for mi in match_ids]
    # run a single metric
    run_metric(graphs, metrics_passing_volume, post_passing_volume)

    """
    # or run all metrics
    for m, pm in zip(metrics, post_metrics):
        run_metric(graphs, m, pm)
    """
