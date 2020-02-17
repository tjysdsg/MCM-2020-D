from soccer_network.data import *
from typing import Tuple, List
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

mpl.rc("savefig", dpi=200)

__all__ = ['get_activity_level']

activity_scores = {
    'Duel': {'Air duel': (0, 1.0, 0, 0),
             'Ground attacking duel': (.15, 0, 0, 0),
             'Ground defending duel': (0, .15, 0, 0),
             'Ground loose ball duel': (.1, 0, 0, 0)
             },
    'Foul': {'Foul': (0, 0, 0, .1),
             'Hand foul': (0, 0, 0, .3),
             'Late card foul': (0, 0, 0, .1),
             'Out of game foul': (0, 0, 0, .3),
             'Protest': (0, 0, 0, 1.),
             'Simulation': (0, 0, 0, .1),
             'Time lost foul': (0, 0, 0, .1),
             'Violent Foul': (.05, 0, 0, .15)
             },
    'Free Kick': {'Corner': (.1, 0, 0, 0),
                  'Free Kick': (.1, 0, 0, 0),
                  'Free kick cross': (.1, 0, .1, 0),
                  'Free kick shot': (.15, 0, 0, 0),
                  'Goal kick': (.2, 0, 0, 0),
                  'Penalty': (.1, 0, 0, 0),
                  'Throw in': (.2, 0, 0, 0)
                  },
    'Goalkeeper leaving line': {'Goalkeeper leaving line': 
        (0, 0, 0, .2)},
    'Interruption': {'Ball out of the field': (0, 0, 0, 0),
    'Whistle': (0, 0, 0, 0)},
    'Offside': (0, 0, 0, .1),
    'Others on the ball': {'Acceleration': (0, 0, 0, 0),
                           'Clearance': (0, .15, 0, 0),
                           'Touch': (0, 0, .15, 0),
                           },
    'Pass': {'Cross': (0, 0, .15, 0),
             'Hand pass': (0, 0, .1, 0),
             'Head pass': (0, 0, .15, 0),
             'High pass': (0, 0, .15, 0),
             'Launch': (0, 0, .1, 0),
             'Simple pass': (0, 0, .1, 0),
             'Smart pass': (0, 0, .2, 0)
             },
    'Save attempt': {'Reflexes': (0, .15, 0, 0),
                     'Save attempt': (0, .1, 0, 0)
                     },
    'Shot': {
        'Shot': (.15, 0, 0, 0)
    }
}


def get_event_scores(etype: str, esubtype: str)
-> Tuple[float, float, float, float]:
    subtype_scores = activity_scores.get(etype)
    if subtype_scores is None:
        return 0, 0, 0, 0
    elif type(subtype_scores) is dict:
        return subtype_scores.get(esubtype, (0, 0, 0, 0))
    else:
        return subtype_scores


def cal_act_lvls(time_series_data: pd.DataFrame,
                 match_id: int or None,
                 player_id: str or None,
                 team_id: str = 'Huskies',
                 interval_seconds: float = 60.0)
                 -> np.ndarray or None:
    timer = 0
    curr_act_lvl = np.zeros(4, dtype=float)
    act_lvls = []
    pcond = True if player_id is None else 
        time_series_data['OriginPlayerID'] == player_id
    mcond = True if match_id is None 
        else time_series_data['MatchID'] == match_id
    data = time_series_data[pcond & mcond &
                            (time_series_data['TeamID'] == team_id)]
    if len(data) == 0:
        print("Warning: no data for player #{0} in match #{1}"
            .format(player_id, match_id))
        return None
    player_time = data['EventTime'].to_numpy()
    etypes = data['EventType']
    esubtypes = data['EventSubType']
    for i in range(data.shape[0] - 1):
        interval = player_time[i + 1] - player_time[i]
        timer += interval
        if timer > interval_seconds:
            data, reset timer
            act_lvls.append(curr_act_lvl.copy())
            timer = interval
            curr_act_lvl.fill(0)
        else:
            curr_act_lvl += np.asarray(get_event_scores(
                etypes.iloc[i], esubtypes.iloc[i]))
            timer += interval
    return np.asarray(act_lvls)


def get_activity_level():
    all_events.loc[all_events['MatchPeriod'] == '2H', 'EventTime']
    += 45 * 60
    all_events.sort_values('EventTime', inplace=True)

    types = ['attack', 'defense', 'collaborate', 'foul']
    df_dict = {'MatchID': match_ids}
    for i in range(4):
        df_dict['huskies_mean_' + types[i]] = []
        df_dict['huskies_std_' + types[i]] = []
        df_dict['oppo_mean_' + types[i]] = []
        df_dict['oppo_std_' + types[i]] = []

    for mi in match_ids:
        match_data = matches_df[matches_df['MatchID'] == mi]
        outcome = match_data['Outcome'].to_list()[0]
        oppo_team_id = match_data['OpponentID'].to_list()[0]

        shots = all_events[(all_events['EventType'] == 'Shot') &
            (all_events['MatchID'] == mi)]
        huskies_shots_time = shots[shots['TeamID'] == 'Huskies']
            ['EventTime']
        oppo_shots_time = shots[shots['TeamID'] == oppo_team_id]
            ['EventTime']

        huskies_act_lvls = cal_act_lvls(all_events, mi, player_id=None,
            team_id='Huskies')
        oppo_act_lvls = cal_act_lvls(all_events, mi, player_id=None,
            team_id=oppo_team_id)

        oppo_len = oppo_act_lvls.shape[0]
        huskies_len = huskies_act_lvls.shape[0]
        if huskies_len < oppo_len:
            huskies_act_lvls = np.pad(huskies_act_lvls, ((0, oppo_len
            - huskies_len), (0, 0)), 'constant')
        else:
            oppo_act_lvls = np.pad(oppo_act_lvls, ((0, huskies_len
            - oppo_len), (0, 0)), 'constant')

        for i in range(4):
            df_dict['huskies_mean_' + types[i]].append(np.mean(
                huskies_act_lvls[:, i]))
            df_dict['huskies_std_' + types[i]].append(np.std(
                huskies_act_lvls[:, i]))
            df_dict['oppo_mean_' + types[i]].append(np.mean(
                oppo_act_lvls[:, i]))
            df_dict['oppo_std_' + types[i]].append(np.std(
                oppo_act_lvls[:, i]))
    df = pd.DataFrame(df_dict)
    df.set_index('MatchID', inplace=True)
    return df
