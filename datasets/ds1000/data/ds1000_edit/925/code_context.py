import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            myData = pd.DataFrame(
                {
                    "Month": [3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
                    "A1": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                    "A2": [31, 13, 13, 13, 33, 33, 81, 38, 18, 38, 18, 18, 118],
                    "A3": [81, 38, 18, 38, 18, 18, 118, 31, 13, 13, 13, 33, 33],
                    "A4": [1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
                }
            )
            scaler = MinMaxScaler()
        return myData, scaler

    def generate_ans(data):
        myData, scaler = data
        cols = myData.columns[2:4]

        def scale(X):
            X_ = np.atleast_2d(X)
            return pd.DataFrame(scaler.fit_transform(X_), X.index)

        myData["new_" + cols] = myData.groupby("Month")[cols].apply(scale)
        return myData

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
myData, scaler = test_input
[insert]
result = myData
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
