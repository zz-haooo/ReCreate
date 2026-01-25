import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["label"] = df.Close.diff().fillna(1).gt(0).astype(int)
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
                    ],
                    "Close": [1460, 1470, 1480, 1450],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "DateTime": [
                        "2010-01-04",
                        "2010-01-05",
                        "2010-01-06",
                        "2010-01-07",
                    ],
                    "Close": [1460, 1470, 1480, 1450],
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
