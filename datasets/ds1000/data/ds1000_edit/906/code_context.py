import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            df1 = pd.DataFrame(
                {
                    "Time": [1, 2, 3, 4, 5, 5.5, 6],
                    "A1": [6.64, 6.70, None, 7.15, None, 7.44, 7.62],
                    "A2": [6.82, 6.86, None, 7.26, None, 7.63, 7.86],
                    "A3": [6.79, 6.92, None, 7.26, None, 7.58, 7.71],
                    "B1": [6.70, None, 7.07, 7.19, None, 7.54, None],
                    "B2": [6.95, None, 7.27, None, 7.40, None, None],
                    "B3": [7.02, None, 7.40, None, 7.51, None, None],
                }
            )
        elif test_case_id == 2:
            df1 = pd.DataFrame(
                {
                    "Time": [1, 2, 3, 4, 5, 5.5],
                    "A1": [6.64, 6.70, np.nan, 7.15, np.nan, 7.44],
                    "A2": [6.82, 6.86, np.nan, 7.26, np.nan, 7.63],
                    "A3": [6.79, 6.92, np.nan, 7.26, np.nan, 7.58],
                    "B1": [6.70, np.nan, 7.07, 7.19, np.nan, 7.54],
                    "B2": [6.95, np.nan, 7.27, np.nan, 7.40, np.nan],
                    "B3": [7.02, np.nan, 7.40, 6.95, 7.51, 6.95],
                    "C1": [np.nan, 6.95, np.nan, 7.02, np.nan, 7.02],
                    "C2": [np.nan, 7.02, np.nan, np.nan, 6.95, np.nan],
                    "C3": [6.95, 6.95, 6.95, 6.95, 7.02, 6.95],
                    "D1": [7.02, 7.02, 7.02, 7.02, np.nan, 7.02],
                    "D2": [np.nan, 3.14, np.nan, 9.28, np.nan, np.nan],
                    "D3": [6.95, 6.95, 6.95, 6.95, 6.95, 6.95],
                }
            )
        return df1

    def generate_ans(data):
        df1 = data
        slopes = []
        for col in df1.columns:
            if col == "Time":
                continue
            mask = ~np.isnan(df1[col])
            x = np.atleast_2d(df1.Time[mask].values).T
            y = np.atleast_2d(df1[col][mask].values).T
            reg = LinearRegression().fit(x, y)
            slopes.append(reg.coef_[0])
        slopes = np.array(slopes).reshape(-1)
        return slopes

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
from sklearn.linear_model import LinearRegression
df1 = test_input
[insert]
result = slopes
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
