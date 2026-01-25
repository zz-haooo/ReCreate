import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df = df.set_index("cat")
        res = df.div(df.sum(axis=0), axis=1)
        return res.reset_index()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "cat": ["A", "B", "C"],
                    "val1": [7, 10, 5],
                    "val2": [10, 2, 15],
                    "val3": [0, 1, 6],
                    "val4": [19, 14, 16],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "cat": ["A", "B", "C"],
                    "val1": [1, 1, 4],
                    "val2": [10, 2, 15],
                    "val3": [0, 1, 6],
                    "val4": [19, 14, 16],
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
