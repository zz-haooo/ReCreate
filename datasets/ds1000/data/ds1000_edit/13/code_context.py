import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["datetime"] = df["datetime"].dt.tz_localize(None)
        df.sort_values(by="datetime", inplace=True)
        df["datetime"] = df["datetime"].dt.strftime("%d-%b-%Y %T")
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "datetime": [
                        "2015-12-01 00:00:00-06:00",
                        "2015-12-02 00:01:00-06:00",
                        "2015-12-03 00:00:00-06:00",
                    ]
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "datetime": [
                        "2016-12-02 00:01:00-06:00",
                        "2016-12-01 00:00:00-06:00",
                        "2016-12-03 00:00:00-06:00",
                    ]
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


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "tz_localize" in tokens
