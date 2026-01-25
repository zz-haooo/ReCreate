import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        label = []
        for i in range(len(df) - 1):
            if df.loc[i, "Close"] > df.loc[i + 1, "Close"]:
                label.append(1)
            elif df.loc[i, "Close"] == df.loc[i + 1, "Close"]:
                label.append(0)
            else:
                label.append(-1)
        label.append(1)
        df["label"] = label
        df["DateTime"] = df["DateTime"].dt.strftime("%d-%b-%Y")
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "DateTime": [
                        "2000-01-04",
                        "2000-01-05",
                        "2000-01-06",
                        "2000-01-07",
                        "2000-01-08",
                    ],
                    "Close": [1460, 1470, 1480, 1480, 1450],
                }
            )
            df["DateTime"] = pd.to_datetime(df["DateTime"])
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "DateTime": [
                        "2000-02-04",
                        "2000-02-05",
                        "2000-02-06",
                        "2000-02-07",
                        "2000-02-08",
                    ],
                    "Close": [1460, 1470, 1480, 1480, 1450],
                }
            )
            df["DateTime"] = pd.to_datetime(df["DateTime"])
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
