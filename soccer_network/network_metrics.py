from soccer_network.data import (match_ids, matches_df)
from soccer_network.graphs import load_graphml
from graph_tool import Graph, clustering, correlations, centrality, VertexPropertyMap
import pandas as pd
import numpy as np
from typing import Tuple, List, Callable, Any

"""
flow
dynamics
configuration model
topology
"""


def get_corr(**kwargs):
    kwargs.update({'MatchID': match_ids})
    df = pd.DataFrame(kwargs).set_index('MatchID')
    df = df.join(matches_df[['Outcome', 'OwnScore']], how='left')
    df['ScoreDiff'] = matches_df['Outcome'] - matches_df['OwnScore']
    return df.corr()


def get_mean_std_of_scalar_vertex_properties(vps: List[VertexPropertyMap]):
    data = [np.asarray(r.a) for r in vps]
    data = np.asarray(data)
    if len(data.shape) != 2:  # the length each row in `data` is not the same, must use loop
        mean = np.asarray([np.mean(d) for d in data])
        std = np.asarray([np.std(d) for d in data])
    else:
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
    return mean, std


def post_results_as_vertex_properties(results: List[VertexPropertyMap]):
    mean, std = get_mean_std_of_scalar_vertex_properties(results)
    print(get_corr(mean=mean, std=std))
    return mean, std


def clustering_coefficient(g: Graph):
    return clustering.local_clustering(g, weight=g.edge_properties['weight'], undirected=False)


def passing_volume(g: Graph):
    vs = g.get_vertices()
    total = g.get_total_degrees(vs, g.edge_properties['weight'])
    return np.mean(total), np.std(total)


def post_passing_volume(results: Tuple):
    # from matplotlib import pyplot as plt
    mean, std = zip(
        *results)  # https://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
    print(get_corr(mean=mean, std=std))


def assortativity(g: Graph):
    return correlations.assortativity(g, 'total', g.edge_properties['weight'])


def post_assortativity(results):
    mean, variance = zip(*results)
    print(get_corr(mean=mean, var=variance))


def motifs(g: Graph, k: int = 3):
    return clustering.motif_significance(g, k)


def post_motifs(results: List[Tuple]):
    n_high_z = np.asarray([np.count_nonzero(
        np.abs(np.asarray(z)[::-1]) >= 2.57
    ) for _, z in results])
    print(get_corr(n_high_z=n_high_z))
    return n_high_z


def pagerank_centrality(g: Graph):
    return centrality.pagerank(g, weight=g.edge_properties['weight'])


def closeness_centrality(g: Graph):
    return centrality.closeness(g, weight=g.edge_properties['weight'])


def betweenness_centrality(g: Graph):
    return centrality.betweenness(g, weight=g.edge_properties['weight'])[0]


post_pagerank_centrality = \
    post_closeness_centrality = \
    post_beweenness_centrality = \
    post_clustering_coefficient = \
    post_results_as_vertex_properties

# TODO: add your metrics here, the function should have only one required argument and can return anything you like
metrics: List[Callable] = [
    passing_volume,
    clustering_coefficient,
    assortativity,
    pagerank_centrality,
    closeness_centrality,
    betweenness_centrality,
    motifs,
]

# TODO: add your post-metrics-computation processing function here
# the results of every metric **for all graphs** are passed in as **one** single argument
# NOTE: the order of results is the same as the order of `match_ids`
post_metrics: List[Callable] = [
    post_passing_volume,
    post_clustering_coefficient,
    post_assortativity,
    post_pagerank_centrality,
    post_closeness_centrality,
    post_beweenness_centrality,
    post_motifs,
]


def run_metric(gs: List[Graph], metric: Callable[[Graph], Any], post_metric: Callable):
    print('=' * 25)
    print('\tRunning {0}'.format(metric.__name__))
    print('=' * 25)
    results = [metric(g) for g in gs]
    post_metric(results)


if __name__ == "__main__":
    print('Loading graphml files...')
    graphs = [load_graphml('../graphs/network-{0}.xml'.format(mi)) for mi in match_ids]
    # graphs = [load_graphml('../graphs/zoned-network-{0}.xml'.format(mi)) for mi in match_ids]

    print('Calculating metrics...')
    # run a single metric
    # run_metric(graphs, motifs, post_motifs)

    # or run all metrics
    for m, pm in zip(metrics, post_metrics):
        run_metric(graphs, m, pm)
