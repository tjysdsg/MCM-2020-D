from soccer_network.data import *
from soccer_network.graphs import load_graphml
from graph_tool import clustering, Graph

"""
clustering
motifs
centrality
flow
dynamics
configuration model
topology
"""


def metric_motifs(g: Graph):
    return clustering.motif_significance(g, 3)


if __name__ == "__main__":
    for mi in match_ids:
        g = load_graphml('../graphs/network-{0}.xml'.format(mi))
        motifs, z_scores = metric_motifs(g)
        print(z_scores)
