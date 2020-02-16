import numpy as np
from typing import List, Tuple
from graph_tool.all import Graph


class Network:
    def __init__(self):
        self.g = Graph(directed=True)
        self.player_id_to_vertex = {}
        self.pairs = {}  # player pair: edge
        # property maps for additional information
        self.g.vertex_properties['player_id'] = self.g.new_vertex_property("string")
        self.g.vertex_properties['player_coords'] = self.g.new_vertex_property("vector<float>")
        self.g.vertex_properties['average_player_coords'] = self.g.new_vertex_property("vector<float>")
        self.g.vertex_properties['player_n_coords'] = self.g.new_vertex_property("int")
        self.g.edge_properties['weight'] = self.g.new_edge_property("int")

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
            self.g.vertex_properties['average_player_coords'][v] = np.asarray(
                self.player_coords_pmap[v]) / self.player_n_coords_pmap[v]
        return self.g.vertex_properties['average_player_coords']

    def add_players(self, pids: List[str]):
        n = len(pids)
        vs = list(self.g.add_vertex(n))
        self.player_id_to_vertex.update({pids[i]: vs[i] for i in range(n)})
        for i in range(n):
            self.player_id_pmap[vs[i]] = pids[i]
        return vs

    def add_passes(self, id_pairs: List[Tuple], coords_pairs: List[Tuple], pass_scores=None):
        pairs = [(self.player_id_to_vertex[i1], self.player_id_to_vertex[i2])
                 for i1, i2 in id_pairs]
        # append player coordinates
        n = len(coords_pairs)
        if pass_scores is None:
            pass_scores = [1 for _ in range(n)]

        for i in range(n):
            # remember orig and dest location
            # orig player
            coords = self.player_coords_pmap[pairs[i][0]]
            if len(coords) == 0:
                coords = np.asarray([coords_pairs[i][0], coords_pairs[i][1]])
            else:
                # accumulate
                coords += np.asarray([coords_pairs[i][0], coords_pairs[i][1]])
            self.player_coords_pmap[pairs[i][0]] = coords
            self.player_n_coords_pmap[pairs[i][0]] += 1

            # dest player
            coords = self.player_coords_pmap[pairs[i][1]]
            if len(coords) == 0:
                coords = np.asarray([coords_pairs[i][2], coords_pairs[i][3]])
            else:
                # accumulate
                coords += np.asarray([coords_pairs[i][2], coords_pairs[i][3]])
            self.player_coords_pmap[pairs[i][1]] = coords
            self.player_n_coords_pmap[pairs[i][1]] += 1

            # if the edge exists, increment its weight instead of creating a new edge
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


# TODO: update this to behave like `Network` above
class ZonedNetwork:
    def __init__(self, size: Tuple[int] = (10, 10), field_size: Tuple[int] = (100, 100)):
        self.g = Graph(directed=True)
        self.n_zones = size[0] * size[1]
        self.fwidth = field_size[0]
        self.fheight = field_size[1]
        self.n_rows = size[0]
        self.n_cols = size[1]
        self.row_size: float = self.fheight / self.n_rows
        self.col_size: float = self.fwidth / self.n_cols
        self.g.add_vertex(self.n_zones)

    def get_zone(self, coords: Tuple):
        r = int(coords[1] / self.row_size)
        c = int(coords[0] / self.col_size)
        r = min(self.n_rows - 1, r)
        c = min(self.n_cols - 1, c)
        return self.g.vertex(r * self.n_cols + c)

    def add_passes(self, coords_pairs: List[Tuple]):
        pairs = [(self.get_zone((x1, y1)), self.get_zone((x2, y2)))
                 for x1, y1, x2, y2 in coords_pairs]
        return self.g.add_edge_list(pairs)

    def save(self, file: str):
        self.g.save(file, fmt='graphml')
