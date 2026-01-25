import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        mask = (df.filter(like="Value").abs() > 1).any(axis=1)
        return df[mask]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "A_Name": ["AA", "BB", "CC", "DD", "EE", "FF", "GG"],
                    "B_Detail": ["X1", "Y1", "Z1", "L1", "M1", "N1", "K1"],
                    "Value_B": [1.2, 0.76, 0.7, 0.9, 1.3, 0.7, -2.4],
                    "Value_C": [0.5, -0.7, -1.3, -0.5, 1.8, -0.8, -1.9],
                    "Value_D": [-1.3, 0.8, 2.5, 0.4, -1.3, 0.9, 2.1],
                }
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "A_Name": ["AA", "BB", "CC", "DD", "EE", "FF", "GG"],
                    "B_Detail": ["X1", "Y1", "Z1", "L1", "M1", "N1", "K1"],
                    "Value_B": [1.2, 0.76, 0.7, 0.9, 1.3, 0.7, -2.4],
                    "Value_C": [0.5, -0.7, -1.3, -0.5, 1.8, -0.8, -1.9],
                    "Value_D": [-1.3, 0.8, 2.5, 0.4, -1.3, 0.9, 2.1],
                    "Value_E": [1, 1, 4, -5, -1, -4, 2.1],
                    "Value_F": [-1.9, 2.6, 0.8, 1.7, -1.3, 0.9, 2.1],
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
