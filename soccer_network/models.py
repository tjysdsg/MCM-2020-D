from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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


def kfold_model(x, y, model, params: dict, n_splits: int, random_state: int):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(df):
        m = model(**params, random_state=random_state)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m.fit(X_train, y_train)
        print(m.score(X_test, y_test))


if __name__ == "__main__":
    from soccer_network.graphs import load_graphml
    from soccer_network.network_metrics import metrics, post_metrics, run_metric

    # network metrics
    graphs = [load_graphml('../graphs/network-{0}.xml'.format(mi)) for mi in match_ids]
    data = []
    for m, pm in zip(metrics, post_metrics):
        data.append(run_metric(graphs, m, pm))
    df = list(zip(*data))  # data now is [38 matchs, 7 metrics, 14 vertices (players)]
    for i in range(len(df)):
        df[i] = np.asarray(df[i])
    for i in range(len(df)):
        if df[i].shape[1] < 14:
            df[i] = np.hstack([df[i], np.mean(df[i], axis=1)[:, None]])
    df = np.asarray(df)
    df.shape = df.shape[0], -1
    df = np.nan_to_num(df)

    # huskies_data = []
    # oppo_data = []
    # for mi in match_ids:
    #     oppo_team_id = matches_df[matches_df['MatchID'] == mi]['OpponentID'].to_list()[0]
    #     huskies_act_lvls = cal_act_lvls(all_events, mi, player_id=None, team_id='Huskies')
    #     oppo_act_lvls = cal_act_lvls(all_events, mi, player_id=None, team_id=oppo_team_id)
    #     huskies_data.append(huskies_act_lvls)
    #     oppo_data.append(oppo_act_lvls)

    # huskies_data = pad(huskies_data)
    # huskies_data.shape = *(huskies_data.shape[:-2]), -1
    # oppo_data = pad(oppo_data)
    # oppo_data.shape = *(oppo_data.shape[:-2]), -1

    # df = np.hstack([huskies_data, oppo_data])

    y = matches_df['Outcome']
    # best_params = {'criterion': 'gini', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 80}
    # kfold_model(df, y, RandomForestClassifier, best_params, n_splits=5, random_state=22)

    best_network_params = {'criterion': 'entropy', 'max_depth': 3, 'max_features': 'auto', 'n_estimators': 90}
    kfold_model(df, y, RandomForestClassifier, best_network_params, n_splits=5, random_state=140)

    # parameters = {
    #     'n_estimators': list(range(80, 110, 5)),
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': list(range(2, 4)),
    #     'max_features': ['auto', 'sqrt', 'log2', 0.6],
    # }
    # from sklearn.model_selection import GridSearchCV

    # rfc = RandomForestClassifier()
    # clf = GridSearchCV(rfc, parameters)
    # clf.fit(df, matches_df['Outcome'])
    # print(clf.best_params_)
