import numpy as np
import pandas as pd

__all__ = [
    'matches_df',
    'passings_df',
    'events_df',
    'match_ids',
    'huskies_passes',
    'huskies_events',
    'huskies_player_ids',
    'opponent_player_ids',
    'all_events',
]

data_matches_path = '../data/matches.csv'
data_passings_path = '../data/passingevents.csv'
data_events_path = '../data/fullevents.csv'

matches_df = pd.read_csv(data_matches_path)
passings_df = pd.read_csv(data_passings_path)
events_df = pd.read_csv(data_events_path)


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

huskies_player_ids = ['Huskies_D1', 'Huskies_D10',
                    'Huskies_D2', 'Huskies_D3', 
                    'Huskies_D4', 'Huskies_D5', 
                    'Huskies_D6', 'Huskies_D7', 
                      'Huskies_D8', 'Huskies_D9', 
                      'Huskies_F1', 'Huskies_F2', 
                      'Huskies_F3', 'Huskies_F4',
                      'Huskies_F5', 'Huskies_F6', 
                      'Huskies_G1', 'Huskies_M1', 
                      'Huskies_M10', 'Huskies_M11',
                      'Huskies_M12', 'Huskies_M13', 
                      'Huskies_M2', 'Huskies_M3', 
                      'Huskies_M4', 'Huskies_M5',
                      'Huskies_M6', 'Huskies_M7', 
                      'Huskies_M8', 'Huskies_M9']

match_ids = np.unique(matches_df['MatchID'])
