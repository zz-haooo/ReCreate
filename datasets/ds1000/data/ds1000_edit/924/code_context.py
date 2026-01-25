import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Month": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                    "X1": [12, 10, 100, 55, 65, 60, 35, 25, 10, 15, 30, 40, 50],
                    "X2": [10, 15, 24, 32, 8, 6, 10, 23, 24, 56, 45, 10, 56],
                    "X3": [12, 90, 20, 40, 10, 15, 30, 40, 60, 42, 2, 4, 10],
                }
            )
            scaler = MinMaxScaler()
        return df, scaler

    def generate_ans(data):
        df, scaler = data
        cols = df.columns[2:4]

        def scale(X):
            X_ = np.atleast_2d(X)
            return pd.DataFrame(scaler.fit_transform(X_), X.index)

        df[cols + "_scale"] = df.groupby("Month")[cols].apply(scale)
        return df

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
df, scaler = test_input
[insert]
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
