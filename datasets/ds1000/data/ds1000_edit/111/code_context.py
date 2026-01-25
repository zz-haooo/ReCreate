import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["state"] = np.where(
            (df["col2"] > 50) & (df["col3"] > 50),
            df["col1"],
            df[["col1", "col2", "col3"]].sum(axis=1),
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "datetime": [
                        "2021-04-10 01:00:00",
                        "2021-04-10 02:00:00",
                        "2021-04-10 03:00:00",
                        "2021-04-10 04:00:00",
                        "2021-04-10 05:00:00",
                    ],
                    "col1": [25, 25, 25, 50, 100],
                    "col2": [50, 50, 100, 50, 100],
                    "col3": [50, 50, 50, 100, 100],
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "datetime": [
                        "2021-04-10 01:00:00",
                        "2021-04-10 02:00:00",
                        "2021-04-10 03:00:00",
                        "2021-04-10 04:00:00",
                        "2021-04-10 05:00:00",
                    ],
                    "col1": [25, 10, 66, 50, 100],
                    "col2": [50, 13, 100, 50, 100],
                    "col3": [50, 16, 50, 100, 100],
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
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
