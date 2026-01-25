import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.datasets import make_regression


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_regression(
                n_samples=100,
                n_features=1,
                n_informative=1,
                bias=150.0,
                noise=30,
                random_state=42,
            )
        return X.flatten(), y, X

    def generate_ans(data):
        X, y, X_test = data
        regressor = RandomForestRegressor(
            n_estimators=150, min_samples_split=1.0, random_state=42
        )
        regressor.fit(X.reshape(-1, 1), y)
        predict = regressor.predict(X_test)
        return predict

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
from sklearn.ensemble import RandomForestRegressor
X, y, X_test = test_input
[insert]
predict = regressor.predict(X_test)
result = predict
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
