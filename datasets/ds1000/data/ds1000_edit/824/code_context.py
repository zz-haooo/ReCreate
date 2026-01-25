import pandas as pd
import copy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Col1": ["C", "A", "B"],
                    "Col2": ["33", "2.5", "42"],
                    "Col3": [
                        ["Apple", "Orange", "Banana"],
                        ["Apple", "Grape"],
                        ["Banana"],
                    ],
                }
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Col1": ["c", "a", "b"],
                    "Col2": ["3", "2", "4"],
                    "Col3": [
                        ["Apple", "Orange", "Banana"],
                        ["Apple", "Grape"],
                        ["Banana"],
                    ],
                }
            )
        elif test_case_id == 3:
            df = pd.DataFrame(
                {
                    "Col1": ["c", "a", "b"],
                    "Col2": ["3", "2", "4"],
                    "Col3": [
                        ["Apple", "Orange", "Banana", "Watermelon"],
                        ["Apple", "Grape"],
                        ["Banana"],
                    ],
                }
            )
        elif test_case_id == 4:
            df = pd.DataFrame(
                {
                    "Col1": ["C", "A", "B", "D"],
                    "Col2": ["33", "2.5", "42", "666"],
                    "Col3": ["11", "4.5", "14", "1919810"],
                    "Col4": [
                        ["Apple", "Orange", "Banana"],
                        ["Apple", "Grape"],
                        ["Banana"],
                        ["Suica", "Orange"],
                    ],
                }
            )
        return df

    def generate_ans(data):
        df = data
        mlb = MultiLabelBinarizer()
        df_out = df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop(df.columns[-1])),
                index=df.index,
                columns=mlb.classes_,
            )
        )
        return df_out

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        for i in range(2):
            pd.testing.assert_series_equal(
                result.iloc[:, i], ans.iloc[:, i], check_dtype=False
            )
    except:
        return 0
    try:
        for c in ans.columns:
            pd.testing.assert_series_equal(result[c], ans[c], check_dtype=False)
        return 1
    except:
        pass
    try:
        for c in ans.columns:
            ans[c] = ans[c].replace(1, 2).replace(0, 1).replace(2, 0)
            pd.testing.assert_series_equal(result[c], ans[c], check_dtype=False)
        return 1
    except:
        pass
    return 0


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
df = test_input
[insert]
result = df_out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(4):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
