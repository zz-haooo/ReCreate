import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df[["time", "number"]] = df.duration.str.extract(r"\s*(.*)(\d+)", expand=True)
        for i in df.index:
            df.loc[i, "time"] = df.loc[i, "time"].strip()
        df["time_days"] = df["time"].replace(
            ["year", "month", "week", "day"], [365, 30, 7, 1], regex=True
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {"duration": ["year 7", "day2", "week 4", "month 8"]},
                index=list(range(1, 5)),
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {"duration": ["year 2", "day6", "week 8", "month 7"]},
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
