from soccer_network.data import *
from typing import Tuple, List
import pandas as pd
import numpy as np

"""
See also data.md
- Subtype of Duel: ['Air duel', 'Ground attacking duel', 'Ground defending duel', 'Ground loose ball duel']
'Foul': ['Foul', 'Hand foul', 'Late card foul', 'Out of game foul', 'Protest', 'Simulation', 'Time lost foul', 'Violent Foul']
'Free Kick': ['Corner', 'Free Kick', 'Free kick cross', 'Free kick shot', 'Goal kick', 'Penalty', 'Throw in']
'Goalkeeper leaving line': ['Goalkeeper leaving line']
'Interruption': ['Ball out of the field', 'Whistle']
'Offside': [nan]
'Others on the ball': ['Acceleration', 'Clearance', 'Touch']
'Pass': ['Cross', 'Hand pass', 'Head pass', 'High pass', 'Launch', 'Simple pass', 'Smart pass']
'Save attempt': ['Reflexes', 'Save attempt']
'Shot': ['Shot']
'Substitution': ['Substitution']
"""

# (attack, defense, collaborate, foul)
activity_scores = {
    'Duel': {'Air duel': (.2, 0, 0, 0),
             'Ground attacking duel': (.15, 0, 0, 0),
             'Ground defending duel': (0, .15, 0, 0),
             'Ground loose ball duel': (.1, 0, 0, 0)
             },
    'Foul': {'Foul': (0, 0, 0, .1),
             'Hand foul': (0, 0, 0, .3),
             'Late card foul': (0, 0, 0, .1),
             'Out of game foul': (0, 0, 0, .3),
             'Protest': (0, 0, 0, 1.),  # 不服裁判？抗议？
             'Simulation': (0, 0, 0, .1),
             'Time lost foul': (0, 0, 0, .1),
             'Violent Foul': (.05, 0, 0, .15)
             },
    'Free Kick': {'Corner': (.1, 0, 0, 0),  # 角球
                  'Free Kick': (.1, 0, 0, 0),  # 任意球
                  'Free kick cross': (.1, 0, .1, 0),  # 长传任意球？
                  'Free kick shot': (.15, 0, 0, 0),  # 射门任意球
                  'Goal kick': (.2, 0, 0, 0),  # 任意球进球
                  'Penalty': (.1, 0, 0, 0),  # 禁区内任意球？
                  'Throw in': (.2, 0, 0, 0)  # 手扔着发界外球？
                  },
    'Goalkeeper leaving line': {'Goalkeeper leaving line': (0, 0, 0, .2)
                                # Goalkeepers who leave their goal-line too early at a penalty kick should
                                # get a yellow card and the kick needs to be retaken
                                },
    'Interruption': {'Ball out of the field': (0, 0, 0, 0), 'Whistle': (0, 0, 0, 0)},
    'Offside': (0, 0, 0, .1),  # 越位
    'Others on the ball': {'Acceleration': (0, 0, 0, 0),
                           'Clearance': (0, .15, 0, 0),  # 解围
                           'Touch': (0, 0, .15, 0),  # 快传？
                           },
    'Pass': {'Cross': (0, 0, .15, 0),  # 传中
             'Hand pass': (0, 0, .1, 0),
             'Head pass': (0, 0, .15, 0),
             'High pass': (0, 0, .15, 0),
             'Launch': (0, 0, .1, 0),  # WTF is this?
             'Simple pass': (0, 0, .1, 0),
             'Smart pass': (0, 0, .2, 0)
             },
    'Save attempt': {'Reflexes': (0, .15, 0, 0),  # 快速反应
                     'Save attempt': (0, .1, 0, 0)
                     },
    'Shot': {
        'Shot': (.15, 0, 0, 0)
    }
}


def get_activity_scores(etype: str, esubtype: str) -> Tuple[float, float, float, float]:
    subtype_scores = activity_scores.get(etype)
    if subtype_scores is None:
        return 0, 0, 0, 0
    elif type(subtype_scores) is dict:
        return subtype_scores.get(esubtype, (0, 0, 0, 0))
    else:
        return subtype_scores


def player_activity_levels(time_series_data: pd.DataFrame,
                           match_id: int or None,
                           player_id: str or None,
                           interval_seconds: float = 60.0) -> np.ndarray or None:
    """Calculate the activity levels of a player in a series of time interval"""
    timer = 0
    activity_scores = np.zeros(4, dtype=float)
    act_lvls = []
    pcond = True if player_id is None else time_series_data['OriginPlayerID'] == player_id
    mcond = True if match_id is None else time_series_data['MatchID'] == match_id
    player_data = time_series_data[pcond & mcond]
    if len(player_data) == 0:
        print("Warning: no data for player #{0} in match #{1}".format(player_id, match_id))
        return None
    player_time = player_data['EventTime'].to_numpy()
    etypes = player_data['EventType']
    esubtypes = player_data['EventSubType']
    for i in range(player_data.shape[0] - 1):
        interval = player_time[i + 1] - player_time[i]
        timer += interval
        if timer > interval_seconds:  # if reached 1 min, calc and append new data, reset timer
            act_lvls.append(activity_scores.copy())
            timer = interval
            activity_scores.fill(0)
        else:  # otherwise store data and increment timer
            activity_scores += np.asarray(get_activity_scores(etypes.iloc[i], esubtypes.iloc[i]))
            timer += interval
    return np.asarray(act_lvls)


def plot_act_lvls(data, fig_name: str):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 2)

    x = range(data.shape[0])

    axs[0][0].plot(x, data[:, 0])
    axs[0][0].set_title('Attack')

    axs[0][1].plot(x, data[:, 1])
    axs[0][1].set_title('Defense')

    axs[1][0].plot(x, data[:, 2])
    axs[1][0].set_title('Collaboration')

    axs[1][1].plot(x, data[:, 3])
    axs[1][1].set_title('Foul')
    fig.suptitle(fig_name)
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    # fig.savefig(fig_name)
    plt.show()


def get_corr(**kwargs):
    kwargs.update({'MatchID': match_ids})
    df = pd.DataFrame(kwargs).set_index('MatchID')
    df = df.join(matches_df[['Outcome', 'OwnScore']], how='left')
    df['ScoreDiff'] = matches_df['Outcome'] - matches_df['OwnScore']
    return df.corr()


if __name__ == '__main__':
    # FIXME is the first half really 45 min?
    all_events.loc[all_events['MatchPeriod'] == '2H', 'EventTime'] += 45 * 60  # add 45 minutes
    all_events.sort_values('EventTime', inplace=True)
    for mi in match_ids:
        match_act_lvls = {mi: player_activity_levels(all_events, mi, player_id=None) for mi in match_ids}
        plot_act_lvls(match_act_lvls[mi], 'match #{}'.format(mi))
    # data = np.asarray(
    #     [
    #         [player_activity_levels(all_events, mi, pid) for pid in huskies_player_ids]
    #         for mi in match_ids
    #     ]
    # )
