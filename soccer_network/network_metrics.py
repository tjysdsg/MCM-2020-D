from soccer_network.data import (match_ids, matches_df)
from soccer_network.graphs import load_graphml
from graph_tool import Graph, clustering, correlations, centrality, VertexPropertyMap
import numpy as np
from typing import Tuple, List, Callable, Any


# def post_results_as_vertex_properties2d(results: List[VertexPropertyMap]):
#     return [np.asarray(r.get_2d_array(r.get_graph().get_vertices())).flatten() for r in results]


def post_results_as_vertex_properties(results: List[VertexPropertyMap]):
    return [np.asarray(r.a) for r in results]


def clustering_coefficient(g: Graph):
    return clustering.local_clustering(g, weight=g.edge_properties['weight'], undirected=False)


def passing_volume(g: Graph):
    vs = g.get_vertices()
    return np.asarray(g.get_total_degrees(vs, g.edge_properties['weight']))


def post_passing_volume(results):
    return results


def assortativity(g: Graph):
    return correlations.assortativity(g, 'total', g.edge_properties['weight'])


def post_assortativity(results):
    mean, variance = zip(*results)
    return mean + variance  # list concat


def motifs(g: Graph, k: int = 3):
    return clustering.motif_significance(g, k)


def post_motifs(results: List[Tuple]):
    return [np.asarray(z)[np.abs(np.asarray(z)[::-1]) >= 2.57] for _, z in results]


def pagerank_centrality(g: Graph):
    return centrality.pagerank(g, weight=g.edge_properties['weight'])


def closeness_centrality(g: Graph):
    return centrality.closeness(g, weight=g.edge_properties['weight'])


def betweenness_centrality(g: Graph):
    return centrality.betweenness(g, weight=g.edge_properties['weight'])[0]


def eigenvector_centrality(g: Graph):
    return centrality.eigenvector(g, g.edge_properties['weight'])[1]


def katz_centrality(g: Graph):
    return centrality.katz(g, weight=g.edge_properties['weight'])


def authority_hub_centrality(g: Graph):
    return centrality.hits(g, weight=g.edge_properties['weight'])[1:]


def post_authority_hub_centrality(results):
    authority, hub = zip(*results)
    return post_results_as_vertex_properties(authority), post_results_as_vertex_properties(hub)


post_pagerank_centrality = \
    post_eigenvector_centrality = \
    post_beweenness_centrality = \
    post_closeness_centrality = \
    post_clustering_coefficient = \
    post_katz_centrality = \
    post_results_as_vertex_properties

metrics: List[Callable] = [
    passing_volume,
    clustering_coefficient,
    # assortativity,
    pagerank_centrality,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    katz_centrality,
    # authority_hub_centrality,
    # motifs,
]

# the results of every metric **for all graphs** are passed in as **one** single argument
# NOTE: the order of results is the same as the order of `match_ids`
post_metrics: List[Callable] = [
    post_passing_volume,
    post_clustering_coefficient,
    # post_assortativity,
    post_pagerank_centrality,
    post_closeness_centrality,
    post_beweenness_centrality,
    post_eigenvector_centrality,
    post_katz_centrality,
    # post_authority_hub_centrality,
    # post_motifs,
]


def run_metric(gs: List[Graph], metric: Callable[[Graph], Any], post_metric: Callable) -> List[np.ndarray]:
    results = [metric(g) for g in gs]
    return post_metric(results)


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
