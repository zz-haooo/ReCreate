import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        softmax = []
        min_max = []
        for i in range(len(df)):
            Min = np.inf
            Max = -np.inf
            exp_Sum = 0
            for j in range(len(df)):
                if df.loc[i, "a"] == df.loc[j, "a"]:
                    Min = min(Min, df.loc[j, "b"])
                    Max = max(Max, df.loc[j, "b"])
                    exp_Sum += np.exp(df.loc[j, "b"])
            softmax.append(np.exp(df.loc[i, "b"]) / exp_Sum)
            min_max.append((df.loc[i, "b"] - Min) / (Max - Min))
        df["softmax"] = softmax
        df["min-max"] = min_max
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                    "b": [12, 13, 23, 22, 23, 24, 30, 35, 55],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "a": [4, 4, 4, 5, 5, 5, 6, 6, 6],
                    "b": [12, 13, 23, 22, 23, 24, 30, 35, 55],
                }
            )
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
df = test_input
[insert]
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
