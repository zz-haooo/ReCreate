import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["bar"] = pd.to_numeric(df["bar"], errors="coerce")
        res = df.groupby(["id1", "id2"])[["foo", "bar"]].mean()
        return res

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "foo": [8, 5, 3, 4, 7, 9, 5, 7],
                    "id1": [1, 1, 1, 1, 1, 1, 1, 1],
                    "bar": ["NULL", "NULL", "NULL", 1, 3, 4, 2, 3],
                    "id2": [1, 1, 1, 2, 2, 3, 3, 1],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "foo": [18, 5, 3, 4, 17, 9, 5, 7],
                    "id1": [1, 1, 1, 1, 1, 1, 1, 1],
                    "bar": ["NULL", "NULL", "NULL", 11, 3, 4, 2, 3],
                    "id2": [1, 1, 1, 2, 2, 3, 3, 1],
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
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
