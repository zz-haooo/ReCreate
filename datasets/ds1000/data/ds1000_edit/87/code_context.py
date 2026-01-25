import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df[["number", "time"]] = df.duration.str.extract(r"(\d+)\s*(.*)", expand=True)
        df["time_days"] = df["time"].replace(
            ["year", "month", "week", "day"], [365, 30, 7, 1], regex=True
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {"duration": ["7 year", "2day", "4 week", "8 month"]},
                index=list(range(1, 5)),
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {"duration": ["2 year", "6day", "8 week", "7 month"]},
                index=list(range(1, 5)),
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
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
