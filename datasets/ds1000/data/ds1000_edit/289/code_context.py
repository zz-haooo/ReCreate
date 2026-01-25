import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df_a, df_b = data
        return df_a[["EntityNum", "foo"]].merge(
            df_b[["EntityNum", "a_col"]], on="EntityNum", how="left"
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df_a = pd.DataFrame(
                {"EntityNum": [1001.01, 1002.02, 1003.03], "foo": [100, 50, 200]}
            )
            df_b = pd.DataFrame(
                {
                    "EntityNum": [1001.01, 1002.02, 1003.03],
                    "a_col": ["alice", "bob", "777"],
                    "b_col": [7, 8, 9],
                }
            )
        if test_case_id == 2:
            df_a = pd.DataFrame(
                {"EntityNum": [1001.01, 1002.02, 1003.03], "foo": [100, 50, 200]}
            )
            df_b = pd.DataFrame(
                {
                    "EntityNum": [1001.01, 1002.02, 1003.03],
                    "a_col": ["666", "bob", "alice"],
                    "b_col": [7, 8, 9],
                }
            )
        return df_a, df_b

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
df_a, df_b = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
