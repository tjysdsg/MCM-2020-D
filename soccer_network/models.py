from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from soccer_network.activity_level import get_activity_level
from soccer_network.data import *

if __name__ == "__main__":
    df = get_activity_level()
    rfc = RandomForestClassifier(random_state=0)
    # rfc.fit(df.iloc[:37], matches_df['Outcome'].iloc[:37])

    parameters = {
        'n_estimators': list(range(80, 110, 5)),
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 4)),
        'max_features': ['auto', 'sqrt', 'log2', 0.6],
    }
    clf = GridSearchCV(rfc, parameters)
    clf.fit(df, matches_df['Outcome'])
    print(sorted(clf.cv_results_))
