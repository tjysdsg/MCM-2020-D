from soccer_network.data import *
from typing import Tuple, List
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

mpl.rc("savefig", dpi=200)

__all__ = ['get_activity_level']

# (attack, defense, collaborate, foul)
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


def get_event_scores(etype: str, esubtype: str) -> Tuple[float, float, float, float]:
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
                 interval_seconds: float = 60.0) -> np.ndarray or None:
    """Calculate the activity levels of a player in a series of time interval"""
    timer = 0
    curr_act_lvl = np.zeros(4, dtype=float)
    act_lvls = []
    pcond = True if player_id is None else time_series_data['OriginPlayerID'] == player_id
    mcond = True if match_id is None else time_series_data['MatchID'] == match_id
    data = time_series_data[pcond & mcond &
                            (time_series_data['TeamID'] == team_id)]
    if len(data) == 0:
        print("Warning: no data for player #{0} in match #{1}".format(player_id, match_id))
        return None
    player_time = data['EventTime'].to_numpy()
    etypes = data['EventType']
    esubtypes = data['EventSubType']
    for i in range(data.shape[0] - 1):
        # TODO: special cases
        # passing failed
        # duel failed
        interval = player_time[i + 1] - player_time[i]
        timer += interval
        if timer > interval_seconds:  # if reached 1 min, calc and append new data, reset timer
            act_lvls.append(curr_act_lvl.copy())
            timer = interval
            curr_act_lvl.fill(0)
        else:  # otherwise store data and increment timer
            curr_act_lvl += np.asarray(get_event_scores(etypes.iloc[i], esubtypes.iloc[i]))
            timer += interval
    return np.asarray(act_lvls)


def plot_act_lvls(data, axs, label: str):
    x = range(data.shape[0])

    axs[0][0].plot(x, data[:, 0], label=label)
    axs[0][0].set_title('Attack', loc='left')

    axs[0][1].plot(x, data[:, 1], label=label)
    axs[0][1].set_title('Defense', loc='right')

    axs[1][0].plot(x, data[:, 2], label=label)
    axs[1][0].set_title('Collaboration')

    axs[1][1].plot(x, data[:, 3], label=label)
    axs[1][1].set_title('Foul')


def vline(xs: List[float], ax, color: str, low: float, label: str = None):
    """Plot a vertical line at x"""
    y_vals = [low, 0]
    for x in xs:
        ax.plot([x, x], y_vals, '-', color=color, label=label)


def get_activity_level():
    # FIXME is the first half really 45 min?
    all_events.loc[all_events['MatchPeriod'] == '2H', 'EventTime'] += 45 * 60  # add 45 minutes
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

        shots = all_events[(all_events['EventType'] == 'Shot') & (all_events['MatchID'] == mi)]
        huskies_shots_time = shots[shots['TeamID'] == 'Huskies']['EventTime']
        oppo_shots_time = shots[shots['TeamID'] == oppo_team_id]['EventTime']

        huskies_act_lvls = cal_act_lvls(all_events, mi, player_id=None, team_id='Huskies')
        oppo_act_lvls = cal_act_lvls(all_events, mi, player_id=None, team_id=oppo_team_id)

        oppo_len = oppo_act_lvls.shape[0]
        huskies_len = huskies_act_lvls.shape[0]
        if huskies_len < oppo_len:
            huskies_act_lvls = np.pad(huskies_act_lvls, ((0, oppo_len - huskies_len), (0, 0)), 'constant')
        else:
            oppo_act_lvls = np.pad(oppo_act_lvls, ((0, huskies_len - oppo_len), (0, 0)), 'constant')

        for i in range(4):
            df_dict['huskies_mean_' + types[i]].append(np.mean(huskies_act_lvls[:, i]))
            df_dict['huskies_std_' + types[i]].append(np.std(huskies_act_lvls[:, i]))
            df_dict['oppo_mean_' + types[i]].append(np.mean(oppo_act_lvls[:, i]))
            df_dict['oppo_std_' + types[i]].append(np.std(oppo_act_lvls[:, i]))

        # plot
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(10.24, 7.68)
        plot_act_lvls(huskies_act_lvls, axs, 'Huskies')
        plot_act_lvls(oppo_act_lvls, axs, oppo_team_id)
        # plot shots occurring time
        huskies_shots_time /= 60
        oppo_shots_time /= 60
        low = [
            axs[0, 0].get_ylim()[0],
            axs[0, 1].get_ylim()[0],
            axs[1, 0].get_ylim()[0],
            axs[1, 1].get_ylim()[0],
        ]
        color = 'b'
        name = 'huskies shots'
        for s in [huskies_shots_time, oppo_shots_time]:
            vline(s, axs[0][0], color=color, low=low[0], label=name)
            vline(s, axs[0][1], color=color, low=low[1])
            vline(s, axs[1][0], color=color, low=low[2])
            vline(s, axs[1][1], color=color, low=low[3])
            color = 'r'
            name = 'opponenet shots'
        # configure figures
        fig_name = "match-{}-huskies-vs-{}-outcome-{}".format(mi, oppo_team_id, outcome)
        fig.suptitle(fig_name)
        fig.subplots_adjust(top=0.8)
        fig.tight_layout()
        plt.legend()
        # save and close
        fig.savefig('../images/activity_levels/' + fig_name)
        plt.clf()
        plt.cla()
        plt.close()

    df = pd.DataFrame(df_dict)
    df.set_index('MatchID', inplace=True)
    return df


if __name__ == '__main__':
    get_activity_level()
