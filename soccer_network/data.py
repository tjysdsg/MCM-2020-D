import numpy as np
import pandas as pd
from typing import List

data_matches_path = '../data/matches.csv'
data_passings_path = '../data/passingevents.csv'
data_events_path = '../data/fullevents.csv'

# load data
matches_df = pd.read_csv(data_matches_path)
passings_df = pd.read_csv(data_passings_path)
events_df = pd.read_csv(data_events_path)


# convert 'win', 'tie' and 'lose' to int
def outcome_int_map(x: str):
    if x == 'win':
        return 1
    elif x == 'tie':
        return 0
    else:
        return -1


matches_df['Outcome'] = matches_df['Outcome'].apply(outcome_int_map)

all_events = events_df.join(
    matches_df[['OwnScore', 'OpponentScore', 'Outcome']],
    on='MatchID',
    how='outer')
huskies_passes = all_events[(all_events['TeamID'] == 'Huskies')
                            & (all_events['EventType'] == 'Pass')]
huskies_events = all_events[all_events['TeamID'] == 'Huskies']

huskies_player_ids: List[str] = huskies_events['OriginPlayerID'].tolist() + huskies_events[
    'DestinationPlayerID'].tolist()
huskies_player_ids = np.unique(huskies_player_ids).tolist()
huskies_player_ids.remove('nan')

match_ids = np.unique(matches_df['MatchID'])
