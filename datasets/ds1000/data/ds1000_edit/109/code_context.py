import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df1, df2 = data
        return pd.merge_asof(df1, df2, on="Timestamp", direction="forward")

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df1 = pd.DataFrame(
                {
                    "Timestamp": [
                        "2019/04/02 11:00:01",
                        "2019/04/02 11:00:15",
                        "2019/04/02 11:00:29",
                        "2019/04/02 11:00:30",
                    ],
                    "data": [111, 222, 333, 444],
                }
            )
            df2 = pd.DataFrame(
                {
                    "Timestamp": [
                        "2019/04/02 11:00:14",
                        "2019/04/02 11:00:15",
                        "2019/04/02 11:00:16",
                        "2019/04/02 11:00:30",
                        "2019/04/02 11:00:31",
                    ],
                    "stuff": [101, 202, 303, 404, 505],
                }
            )
            df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
            df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
        if test_case_id == 2:
            df1 = pd.DataFrame(
                {
                    "Timestamp": [
                        "2019/04/02 11:00:01",
                        "2019/04/02 11:00:15",
                        "2019/04/02 11:00:29",
                        "2019/04/02 11:00:30",
                    ],
                    "data": [101, 202, 303, 404],
                }
            )
            df2 = pd.DataFrame(
                {
                    "Timestamp": [
                        "2019/04/02 11:00:14",
                        "2019/04/02 11:00:15",
                        "2019/04/02 11:00:16",
                        "2019/04/02 11:00:30",
                        "2019/04/02 11:00:31",
                    ],
                    "stuff": [111, 222, 333, 444, 555],
                }
            )
            df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
            df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
        return df1, df2

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
df1, df2 = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens and "while" not in tokens
