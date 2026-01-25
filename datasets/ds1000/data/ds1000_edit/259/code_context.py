import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["TIME"] = pd.to_datetime(df["TIME"])
        df["RANK"] = df.groupby("ID")["TIME"].rank(ascending=True)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "ID": ["01", "01", "01", "02", "02"],
                    "TIME": [
                        "2018-07-11 11:12:20",
                        "2018-07-12 12:00:23",
                        "2018-07-13 12:00:00",
                        "2019-09-11 11:00:00",
                        "2019-09-12 12:00:00",
                    ],
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
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
