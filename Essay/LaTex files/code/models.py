def kfold_model(x, y, model, params: dict, n_splits: int, random_state: int):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(df):
        m = model(**params, random_state=random_state)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m.fit(X_train, y_train)
        print(m.score(X_test, y_test))
