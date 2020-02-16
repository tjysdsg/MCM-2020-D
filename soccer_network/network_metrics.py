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


def clustering_coefficient(g: Graph):
    return clustering.local_clustering(g, weight=g.edge_properties['weight'], undirected=False)


def post_clustering_coefficient(results):
    cc = [r.get_array() for r in results]
    cc = np.asarray(cc)
    mean = np.mean(cc, axis=1)
    std = np.std(cc, axis=1)
    df = pd.DataFrame(dict(MatchID=match_ids, cc_mean=mean, cc_sigma=std))
    df = matches_df[['Outcome', 'OwnScore']].join(df, how='right')
    print(df.corr())
    return mean, std


def passing_volume(g: Graph):
    vs = g.get_vertices()
    total = g.get_total_degrees(vs, g.edge_properties['weight'])
    return np.mean(total), np.std(total)


def post_passing_volume(results: Tuple):
    # from matplotlib import pyplot as plt
    mean, sigma = zip(
        *results)  # https://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
    df = pd.DataFrame(dict(MatchID=match_ids, pv_mean=mean, pv_sigma=sigma))
    df = matches_df[['Outcome', 'OwnScore']].join(df, how='right')
    print(df.corr())


def motifs(g: Graph, k: int = 3):
    return clustering.motif_significance(g, k)


def post_motifs(results: List[Tuple]):
    sorted_zs = []
    for mn, z in results:
        z = np.asarray(z)
        z[::-1].sort()
        sorted_zs.append(z)
    print(sorted_zs)
    return sorted_zs


# TODO: add your metrics here, the function should have only one required argument and can return anything you like
metrics = [
    passing_volume,
    clustering_coefficient,
    motifs
]

# TODO| add your post-metrics-computation processing function here,
# the results of every metric **for all graphs** are passed in as **one** single argument
# NOTE: the order of results is the same as the order of `match_ids`
post_metrics = [post_passing_volume,
                post_clustering_coefficient,
                post_motifs]


def run_metric(gs: List[Graph], metric: Callable[[Graph], Any], post_metric: Callable):
    print('Running {0}'.format(metric.__name__))
    results = [metric(g) for g in gs]
    print('Running {0}'.format(post_metric.__name__))
    post_metric(results)


if __name__ == "__main__":
    print('Loading graphml files...')
    graphs = [load_graphml('../graphs/network-{0}.xml'.format(mi)) for mi in match_ids]
    # graphs = [load_graphml('../graphs/zoned-network-{0}.xml'.format(mi)) for mi in match_ids]

    print('Calculating metrics...')
    # run a single metric
    # run_metric(graphs, clustering_coefficient, post_clustering_coefficient)

    # or run all metrics
    for m, pm in zip(metrics, post_metrics):
        run_metric(graphs, m, pm)
