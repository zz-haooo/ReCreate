import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, X = data
        t = df["date"]
        df["date"] = pd.to_datetime(df["date"])
        filter_ids = [0]
        last_day = df.loc[0, "date"]
        for index, row in df[1:].iterrows():
            if (row["date"] - last_day).days > X:
                filter_ids.append(index)
                last_day = row["date"]
        df["date"] = t
        return df.loc[filter_ids, :]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "ID": [1, 2, 3, 4, 5, 6, 7, 8],
                    "date": [
                        "09/15/07",
                        "06/01/08",
                        "10/25/08",
                        "1/14/9",
                        "05/13/09",
                        "11/07/09",
                        "11/15/09",
                        "07/03/11",
                    ],
                    "close": [
                        123.45,
                        130.13,
                        132.01,
                        118.34,
                        514.14,
                        145.99,
                        146.73,
                        171.10,
                    ],
                }
            )
            X = 120
        return df, X

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
df, X = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
