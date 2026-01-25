import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        F = {}
        cnt = 0
        for i in range(len(df)):
            if df["name"].iloc[i] not in F.keys():
                cnt += 1
                F[df["name"].iloc[i]] = cnt
            df.loc[i, "name"] = F[df.loc[i, "name"]]
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "name": ["Aaron", "Aaron", "Aaron", "Brave", "Brave", "David"],
                    "a": [3, 3, 3, 4, 3, 5],
                    "b": [5, 6, 6, 6, 6, 1],
                    "c": [7, 9, 10, 0, 1, 4],
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
def f(df):
[insert]
df = test_input
result = f(df)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
