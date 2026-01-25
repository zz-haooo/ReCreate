import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            X = np.array([[-1, 2], [-0.5, 6]])
        return X

    def generate_ans(data):
        X = data
        scaler = MinMaxScaler()
        X_one_column = X.reshape([-1, 1])
        result_one_column = scaler.fit_transform(X_one_column)
        result = result_one_column.reshape(X.shape)
        return result

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
from sklearn.preprocessing import MinMaxScaler
np_array = test_input
[insert]
result = transformed
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
