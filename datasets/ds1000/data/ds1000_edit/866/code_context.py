import numpy as np
import pandas as pd
import copy
import tokenize, io
from sklearn import datasets


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            iris = datasets.load_iris()
            X = iris.data[(iris.target == 0) | (iris.target == 1)]
            Y = iris.target[(iris.target == 0) | (iris.target == 1)]
            train_indices = list(range(40)) + list(range(50, 90))
            test_indices = list(range(40, 50)) + list(range(90, 100))
            X_train = X[train_indices]
            y_train = Y[train_indices]
            X_train = pd.DataFrame(X_train)
        return X_train, y_train

    def generate_ans(data):
        X_train, y_train = data
        X_train[0] = ["a"] * 40 + ["b"] * 40
        catVar = pd.get_dummies(X_train[0]).to_numpy()
        X_train = np.concatenate((X_train.iloc[:, 1:], catVar), axis=1)
        return X_train

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        if type(result) == np.ndarray:
            np.testing.assert_equal(ans[:, :3], result[:, :3])
        elif type(result) == pd.DataFrame:
            np.testing.assert_equal(ans[:, :3], result.to_numpy()[:, :3])
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
X_train, y_train = test_input
X_train[0] = ['a'] * 40 + ['b'] * 40
[insert]
clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=50).fit(X_train, y_train)
result = X_train
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "get_dummies" in tokens and "OneHotEncoder" not in tokens
