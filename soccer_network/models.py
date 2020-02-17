from sklearn.ensemble import RandomForestClassifier
from soccer_network.activity_level import cal_act_lvls
from soccer_network.data import *
import numpy as np


def pad(data):
    lens = np.array([i.shape[0] for i in data])
    max_len = np.max(lens)
    for i in range(len(data)):
        if data[i].shape[0] < max_len:
            pad_len = max_len - data[i].shape[0]
            data[i] = np.vstack([data[i], np.zeros((pad_len, data[i].shape[1]))])
    return np.asarray(data)


if __name__ == "__main__":
    huskies_data = []
    oppo_data = []
    for mi in match_ids:
        oppo_team_id = matches_df[matches_df['MatchID'] == mi]['OpponentID'].to_list()[0]
        huskies_act_lvls = cal_act_lvls(all_events, mi, player_id=None, team_id='Huskies')
        oppo_act_lvls = cal_act_lvls(all_events, mi, player_id=None, team_id=oppo_team_id)
        huskies_data.append(huskies_act_lvls)
        oppo_data.append(oppo_act_lvls)

    huskies_data = pad(huskies_data)
    oppo_data = pad(oppo_data)

    df = np.hstack([huskies_data, oppo_data])

    y = matches_df['Outcome']
    best_params = {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'auto', 'n_estimators': 80}
    # rfc = RandomForestClassifier(**best_params)
    # rfc.fit(df, y)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(df):
        print("TRAIN:", train_index, "TEST:", test_index)
        rfc = RandomForestClassifier(**best_params)
        X_train, X_test = df[train_index], df[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        print(rfc.score(X_test, y_test))

    # parameters = {
    #     'n_estimators': list(range(80, 110, 5)),
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': list(range(2, 4)),
    #     'max_features': ['auto', 'sqrt', 'log2', 0.6],
    # }
    # from sklearn.model_selection import GridSearchCV
    # clf = GridSearchCV(rfc, parameters)
    # clf.fit(df, matches_df['Outcome'])
    # print(clf.best_params_)
