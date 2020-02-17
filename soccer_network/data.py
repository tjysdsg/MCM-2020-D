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

huskies_player_ids = ['Huskies_D1', 'Huskies_D10', 'Huskies_D2', 'Huskies_D3', 'Huskies_D4', 'Huskies_D5', 'Huskies_D6',
                      'Huskies_D7', 'Huskies_D8', 'Huskies_D9', 'Huskies_F1', 'Huskies_F2', 'Huskies_F3', 'Huskies_F4',
                      'Huskies_F5', 'Huskies_F6', 'Huskies_G1', 'Huskies_M1', 'Huskies_M10', 'Huskies_M11',
                      'Huskies_M12', 'Huskies_M13', 'Huskies_M2', 'Huskies_M3', 'Huskies_M4', 'Huskies_M5',
                      'Huskies_M6', 'Huskies_M7', 'Huskies_M8', 'Huskies_M9']

opponent_player_ids = ['Opponent10_D1', 'Opponent10_D2', 'Opponent10_D3', 'Opponent10_D4',
                       'Opponent10_D5', 'Opponent10_D6', 'Opponent10_F1', 'Opponent10_F2', 'Opponent10_F3',
                       'Opponent10_F4', 'Opponent10_G1', 'Opponent10_M1', 'Opponent10_M2', 'Opponent10_M3',
                       'Opponent10_M4', 'Opponent10_M5', 'Opponent10_M6', 'Opponent10_M7', 'Opponent11_D1',
                       'Opponent11_D2', 'Opponent11_D3', 'Opponent11_D4', 'Opponent11_D5', 'Opponent11_D6',
                       'Opponent11_D7', 'Opponent11_F1', 'Opponent11_F2', 'Opponent11_F3', 'Opponent11_F4',
                       'Opponent11_F5', 'Opponent11_G1', 'Opponent11_G2', 'Opponent11_M1', 'Opponent11_M2',
                       'Opponent11_M3', 'Opponent11_M4', 'Opponent11_M5', 'Opponent11_M6', 'Opponent12_D1',
                       'Opponent12_D2', 'Opponent12_D3', 'Opponent12_D4', 'Opponent12_D5', 'Opponent12_D6',
                       'Opponent12_D7', 'Opponent12_F1', 'Opponent12_F2', 'Opponent12_G1', 'Opponent12_G2',
                       'Opponent12_M1', 'Opponent12_M2', 'Opponent12_M3', 'Opponent12_M4', 'Opponent12_M5',
                       'Opponent12_M6', 'Opponent12_M7', 'Opponent13_D1', 'Opponent13_D2', 'Opponent13_D3',
                       'Opponent13_D4', 'Opponent13_D5', 'Opponent13_D6', 'Opponent13_F1', 'Opponent13_F2',
                       'Opponent13_G1', 'Opponent13_G2', 'Opponent13_M1', 'Opponent13_M2', 'Opponent13_M3',
                       'Opponent13_M4', 'Opponent13_M5', 'Opponent13_M6', 'Opponent13_M7', 'Opponent13_M8',
                       'Opponent14_D1', 'Opponent14_D2', 'Opponent14_D3', 'Opponent14_D4', 'Opponent14_D5',
                       'Opponent14_D6', 'Opponent14_D7', 'Opponent14_F1', 'Opponent14_F2', 'Opponent14_F3',
                       'Opponent14_F4', 'Opponent14_G1', 'Opponent14_G2', 'Opponent14_M1', 'Opponent14_M2',
                       'Opponent14_M3', 'Opponent14_M4', 'Opponent14_M5', 'Opponent14_M6', 'Opponent15_D1',
                       'Opponent15_D2', 'Opponent15_D3', 'Opponent15_D4', 'Opponent15_D5', 'Opponent15_D6',
                       'Opponent15_F1', 'Opponent15_F2', 'Opponent15_F3', 'Opponent15_F4', 'Opponent15_G1',
                       'Opponent15_M1', 'Opponent15_M2', 'Opponent15_M3', 'Opponent15_M4', 'Opponent15_M5',
                       'Opponent15_M6', 'Opponent15_M7', 'Opponent15_M8', 'Opponent16_D1', 'Opponent16_D2',
                       'Opponent16_D3', 'Opponent16_D4', 'Opponent16_D5', 'Opponent16_D6', 'Opponent16_F1',
                       'Opponent16_F2', 'Opponent16_F3', 'Opponent16_F4', 'Opponent16_F5', 'Opponent16_F6',
                       'Opponent16_G1', 'Opponent16_G2', 'Opponent16_M1', 'Opponent16_M2', 'Opponent16_M3',
                       'Opponent16_M4', 'Opponent17_D1', 'Opponent17_D2', 'Opponent17_D3', 'Opponent17_D4',
                       'Opponent17_D5', 'Opponent17_F1', 'Opponent17_F2', 'Opponent17_F3', 'Opponent17_F4',
                       'Opponent17_G1', 'Opponent17_G2', 'Opponent17_M1', 'Opponent17_M2', 'Opponent17_M3',
                       'Opponent17_M4', 'Opponent17_M5', 'Opponent17_M6', 'Opponent17_M7', 'Opponent17_M8',
                       'Opponent18_D1', 'Opponent18_D2', 'Opponent18_D3', 'Opponent18_D4', 'Opponent18_F1',
                       'Opponent18_F2', 'Opponent18_F3', 'Opponent18_G1', 'Opponent18_M1', 'Opponent18_M2',
                       'Opponent18_M3', 'Opponent18_M4', 'Opponent18_M5', 'Opponent18_M6', 'Opponent18_M7',
                       'Opponent18_M8', 'Opponent19_D1', 'Opponent19_D2', 'Opponent19_D3', 'Opponent19_D4',
                       'Opponent19_D5', 'Opponent19_F1', 'Opponent19_F2', 'Opponent19_F3', 'Opponent19_G1',
                       'Opponent19_M1', 'Opponent19_M2', 'Opponent19_M3', 'Opponent19_M4', 'Opponent19_M5',
                       'Opponent19_M6', 'Opponent1_D1', 'Opponent1_D2', 'Opponent1_D3', 'Opponent1_D4', 'Opponent1_D5',
                       'Opponent1_D6', 'Opponent1_F1', 'Opponent1_F2', 'Opponent1_F3', 'Opponent1_F4', 'Opponent1_F5',
                       'Opponent1_F6', 'Opponent1_G1', 'Opponent1_M1', 'Opponent1_M2', 'Opponent1_M3', 'Opponent1_M4',
                       'Opponent1_M5', 'Opponent1_M6', 'Opponent2_D1', 'Opponent2_D2', 'Opponent2_D3', 'Opponent2_D4',
                       'Opponent2_D5', 'Opponent2_D6', 'Opponent2_F1', 'Opponent2_F2', 'Opponent2_F3', 'Opponent2_G1',
                       'Opponent2_M1', 'Opponent2_M2', 'Opponent2_M3', 'Opponent2_M4', 'Opponent2_M5', 'Opponent2_M6',
                       'Opponent3_D1', 'Opponent3_D2', 'Opponent3_D3', 'Opponent3_D4', 'Opponent3_D5', 'Opponent3_D6',
                       'Opponent3_F1', 'Opponent3_F2', 'Opponent3_F3', 'Opponent3_F4', 'Opponent3_G1', 'Opponent3_M1',
                       'Opponent3_M2', 'Opponent3_M3', 'Opponent3_M4', 'Opponent3_M5', 'Opponent4_D1', 'Opponent4_D2',
                       'Opponent4_D3', 'Opponent4_D4', 'Opponent4_D5', 'Opponent4_D6', 'Opponent4_F1', 'Opponent4_F2',
                       'Opponent4_G1', 'Opponent4_M1', 'Opponent4_M2', 'Opponent4_M3', 'Opponent4_M4', 'Opponent4_M5',
                       'Opponent4_M6', 'Opponent4_M7', 'Opponent4_M8', 'Opponent5_D1', 'Opponent5_D2', 'Opponent5_D3',
                       'Opponent5_D4', 'Opponent5_D5', 'Opponent5_D6', 'Opponent5_D7', 'Opponent5_D8', 'Opponent5_F1',
                       'Opponent5_F2', 'Opponent5_F3', 'Opponent5_G1', 'Opponent5_M1', 'Opponent5_M2', 'Opponent5_M3',
                       'Opponent5_M4', 'Opponent5_M5', 'Opponent5_M6', 'Opponent5_M7', 'Opponent6_D1', 'Opponent6_D2',
                       'Opponent6_D3', 'Opponent6_D4', 'Opponent6_D5', 'Opponent6_F1', 'Opponent6_F2', 'Opponent6_F3',
                       'Opponent6_F4', 'Opponent6_F5', 'Opponent6_G1', 'Opponent6_M1', 'Opponent6_M2', 'Opponent6_M3',
                       'Opponent6_M4', 'Opponent6_M5', 'Opponent6_M6', 'Opponent6_M7', 'Opponent7_D1', 'Opponent7_D2',
                       'Opponent7_D3', 'Opponent7_D4', 'Opponent7_F1', 'Opponent7_F2', 'Opponent7_G1', 'Opponent7_M1',
                       'Opponent7_M2', 'Opponent7_M3', 'Opponent7_M4', 'Opponent7_M5', 'Opponent7_M6', 'Opponent7_M7',
                       'Opponent8_D1', 'Opponent8_D2', 'Opponent8_D3', 'Opponent8_D4', 'Opponent8_D5', 'Opponent8_D6',
                       'Opponent8_D7', 'Opponent8_F1', 'Opponent8_F2', 'Opponent8_F3', 'Opponent8_F4', 'Opponent8_F5',
                       'Opponent8_G1', 'Opponent8_M1', 'Opponent8_M2', 'Opponent8_M3', 'Opponent8_M4', 'Opponent8_M5',
                       'Opponent8_M6', 'Opponent9_D1', 'Opponent9_D2', 'Opponent9_D3', 'Opponent9_D4', 'Opponent9_D5',
                       'Opponent9_D6', 'Opponent9_F1', 'Opponent9_F2', 'Opponent9_F3', 'Opponent9_G1', 'Opponent9_G2',
                       'Opponent9_M1', 'Opponent9_M2', 'Opponent9_M3', 'Opponent9_M4', 'Opponent9_M5', 'Opponent9_M6']

match_ids = np.unique(matches_df['MatchID'])
