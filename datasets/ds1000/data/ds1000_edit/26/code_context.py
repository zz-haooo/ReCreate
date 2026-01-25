import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["#1"] = np.roll(df["#1"], shift=1)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "#1": [11.6985, 43.6431, 54.9089, 63.1225, 72.4399],
                    "#2": [126.0, 134.0, 130.0, 126.0, 120.0],
                },
                index=[
                    "1980-01-01",
                    "1980-01-02",
                    "1980-01-03",
                    "1980-01-04",
                    "1980-01-05",
                ],
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {"#1": [45, 51, 14, 11, 14], "#2": [126.0, 134.0, 130.0, 126.0, 120.0]},
                index=[
                    "1980-01-01",
                    "1980-01-02",
                    "1980-01-03",
                    "1980-01-04",
                    "1980-01-05",
                ],
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
