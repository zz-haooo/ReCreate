import numpy as np
import copy
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.datasets import make_classification


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_classification(
                n_samples=30, n_features=4, n_redundant=0, random_state=42
            )
        elif test_case_id == 2:
            X, y = make_classification(
                n_samples=30, n_features=4, n_redundant=0, random_state=24
            )
        return X, y

    def generate_ans(data):
        X_train, y_train = data
        X_test = X_train
        param_grid = {
            "estimator__max_depth": [1, 2, 3, 4, 5],
            "max_samples": [0.05, 0.1, 0.2, 0.5],
        }
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        bc = BaggingClassifier(
            dt, n_estimators=20, max_samples=0.5, max_features=0.5, random_state=42
        )
        clf = GridSearchCV(bc, param_grid)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        return proba

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
X_train, y_train = test_input
X_test = X_train
param_grid = {
    'estimator__max_depth': [1, 2, 3, 4, 5],
    'max_samples': [0.05, 0.1, 0.2, 0.5]
}
dt = DecisionTreeClassifier(max_depth=1, random_state=42)
bc = BaggingClassifier(dt, n_estimators=20, max_samples=0.5, max_features=0.5, random_state=42)
[insert]
proba = clf.predict_proba(X_test)
result = proba
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
