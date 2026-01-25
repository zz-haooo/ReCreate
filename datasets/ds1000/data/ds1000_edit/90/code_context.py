import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df1, df2, columns_check_list = data
        mask = (df1[columns_check_list] == df2[columns_check_list]).any(axis=1).values
        return mask

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df1 = pd.DataFrame(
                {
                    "A": [1, 1, 1],
                    "B": [2, 2, 2],
                    "C": [3, 3, 3],
                    "D": [4, 4, 4],
                    "E": [5, 5, 5],
                    "F": [6, 6, 6],
                    "Postset": ["yes", "no", "yes"],
                }
            )
            df2 = pd.DataFrame(
                {
                    "A": [1, 1, 1],
                    "B": [2, 2, 2],
                    "C": [3, 3, 3],
                    "D": [4, 4, 4],
                    "E": [5, 5, 5],
                    "F": [6, 4, 6],
                    "Preset": ["yes", "yes", "yes"],
                }
            )
            columns_check_list = ["A", "B", "C", "D", "E", "F"]
        return df1, df2, columns_check_list

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert (result == ans).all()
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df1, df2, columns_check_list = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
