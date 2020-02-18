from soccer_network.data import (match_ids, matches_df)
from soccer_network.graphs import load_graphml
from graph_tool import Graph, clustering,
    correlations, centrality, VertexPropertyMap
import numpy as np
from typing import Tuple, List, Callable, Any

class Network:
    def __init__(self):
        self.g = Graph(directed=True)
        self.player_id_to_vertex = {}
        self.pairs = {}
        self.g.vertex_properties['player_id'] =
            self.g.new_vertex_property("string")
        self.g.vertex_properties['player_coords'] =
            self.g.new_vertex_property("vector<float>")
        self.g.vertex_properties['average_player_coords'] =
            self.g.new_vertex_property("vector<float>")
        self.g.vertex_properties['player_n_coords'] =
            self.g.new_vertex_property("int")
        self.g.edge_properties['weight'] =
            self.g.new_edge_property("float")

    @property
    def edge_weights(self):
        return self.g.edge_properties['weight']

    @property
    def player_id_pmap(self):
        return self.g.vertex_properties['player_id']

    @property
    def player_coords_pmap(self):
        return self.g.vertex_properties['player_coords']

    @property
    def player_n_coords_pmap(self):
        return self.g.vertex_properties['player_n_coords']

    @property
    def average_player_coords_pmap(self):
        # lazy evaluation of means
        for v in self.g.vertices():
            self.g.vertex_properties['average_player_coords'][v] =
            np.asarray(
                self.player_coords_pmap[v]) /
                self.player_n_coords_pmap[v]
        return self.g.vertex_properties['average_player_coords']

    def add_players(self, pids: List[str]):
        n = len(pids)
        vs = list(self.g.add_vertex(n))
        self.player_id_to_vertex.update(
            {pids[i]: vs[i] for i in range(n)})
        for i in range(n):
            self.player_id_pmap[vs[i]] = pids[i]
        return vs

    def add_passes(self, id_pairs: List[Tuple],
        coords_pairs: List[Tuple], pass_scores=None):
        pairs = [(self.player_id_to_vertex[i1],
            self.player_id_to_vertex[i2])
                 for i1, i2 in id_pairs]
        n = len(coords_pairs)
        if pass_scores is None:
            pass_scores = [1 for _ in range(n)]

        for i in range(n):
            coords = self.player_coords_pmap[pairs[i][0]]
            if len(coords) == 0:
                coords = np.asarray([coords_pairs[i][0],
                    coords_pairs[i][1]])
            else:
                coords += np.asarray([coords_pairs[i][0],
                    coords_pairs[i][1]])
            self.player_coords_pmap[pairs[i][0]] = coords
            self.player_n_coords_pmap[pairs[i][0]] += 1

            coords = self.player_coords_pmap[pairs[i][1]]
            if len(coords) == 0:
                coords = np.asarray([coords_pairs[i][2],
                    coords_pairs[i][3]])
            else:
                coords += np.asarray([coords_pairs[i][2],
                    coords_pairs[i][3]])
            self.player_coords_pmap[pairs[i][1]] = coords
            self.player_n_coords_pmap[pairs[i][1]] += 1

            e = self.pairs.get(pairs[i])
            if e is not None:
                self.edge_weights[e] += pass_scores[i]
            else:
                e = self.g.add_edge(*pairs[i])
                self.pairs[pairs[i]] = e
                self.edge_weights[e] = pass_scores[i]

    def cleanup(self):
        """remove isolated vertices"""
        to_remove = []
        for v in self.g.vertices():
            if v.in_degree() + v.out_degree() == 0:
                to_remove.append(v)
        n = len(to_remove)
        self.g.remove_vertex(to_remove, fast=True)
        print("Removed {0} isolated vertices".format(n))

    def save(self, file: str):
        self.g.save(file, fmt='graphml')


def post_results_as_vertex_properties(results:
        List[VertexPropertyMap]):
    return [np.asarray(r.a) for r in results]


def clustering_coefficient(g: Graph):
    return clustering.local_clustering(g,
        weight=g.edge_properties['weight'], undirected=False)


def assortativity(g: Graph):
    return correlations.assortativity(g, 'total',
        g.edge_properties['weight'])


def post_assortativity(results):
    mean, variance = zip(*results)
    return mean + variance


def pagerank_centrality(g: Graph):
    return centrality.pagerank(g, weight=
        g.edge_properties['weight'])


def closeness_centrality(g: Graph):
    return centrality.closeness(g, weight=
        g.edge_properties['weight'])


def betweenness_centrality(g: Graph):
    return centrality.betweenness(g, weight=
        g.edge_properties['weight'])[0]


def eigenvector_centrality(g: Graph):
    return centrality.eigenvector(g, g.edge_properties['weight'])[1]


def katz_centrality(g: Graph):
    return centrality.katz(g, weight=g.edge_properties['weight'])


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
    pagerank_centrality,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    katz_centrality,
]

post_metrics: List[Callable] = [
    post_passing_volume,
    post_clustering_coefficient,
    post_pagerank_centrality,
    post_closeness_centrality,
    post_beweenness_centrality,
    post_eigenvector_centrality,
    post_katz_centrality,
]


def run_metric(gs: List[Graph], metric:
        Callable[[Graph], Any], post_metric: Callable) -> List[np.ndarray]:
    results = [metric(g) for g in gs]
    return post_metric(results)


if __name__ == "__main__":
    print('Loading graphml files...')
    graphs = [load_graphml('../graphs/network-{0}.xml'.format(mi))
        for mi in match_ids]
    print('Calculating metrics...')
    for m, pm in zip(metrics, post_metrics):
        run_metric(graphs, m, pm)
