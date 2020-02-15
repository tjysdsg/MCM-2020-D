from soccer_network.data import *
from soccer_network.network import Network, ZonedNetwork
from graph_tool import load_graph


def load_graphml(file: str):
    return load_graph(file)


def build_network_graphml(match_id: int):
    data = huskies_passes[huskies_passes['MatchID'] == match_id]
    network = Network()
    id_pairs = [(r['OriginPlayerID'].split('_')[1],
                 r['DestinationPlayerID'].split('_')[1])
                for _, r in data.iterrows()
                if (not pd.isnull(r['OriginPlayerID'])) and (
                    not pd.isnull(r['DestinationPlayerID']))]
    coords_pairs = [(r['EventOrigin_x'], r['EventOrigin_y'],
                     r['EventDestination_x'], r['EventDestination_y'])
                    for _, r in data.iterrows()
                    if (not pd.isnull(r['OriginPlayerID'])) and (
                        not pd.isnull(r['DestinationPlayerID']))]
    network.add_players([pi.split('_')[1] for pi in huskies_player_ids])
    network.add_passes(id_pairs, coords_pairs)
    return network


def build_zoned_network_graphml(match_id: int):
    data = huskies_passes[huskies_passes['MatchID'] == match_id]
    network = ZonedNetwork()
    coords_pairs = [(r['EventOrigin_x'], r['EventOrigin_y'],
                     r['EventDestination_x'], r['EventDestination_y'])
                    for _, r in data.iterrows()
                    if (not pd.isnull(r['OriginPlayerID'])) and (
                        not pd.isnull(r['DestinationPlayerID']))]
    network.add_passes(coords_pairs)
    return network


if __name__ == "__main__":
    for mi in match_ids:
        network = build_network_graphml(mi)
        network.save('../graphs/network-{0}.xml'.format(mi))
        network = build_zoned_network_graphml(mi)
        network.save('../graphs/zoned-network-{0}.xml'.format(mi))
