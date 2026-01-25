import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df1 = df.groupby("Date").agg(lambda x: x.eq(0).sum())
        df2 = df.groupby("Date").agg(lambda x: x.ne(0).sum())
        return df1, df2

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Date": ["20.07.2018", "20.07.2018", "21.07.2018", "21.07.2018"],
                    "B": [10, 1, 0, 1],
                    "C": [8, 0, 1, 0],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Date": ["20.07.2019", "20.07.2019", "21.07.2019", "21.07.2019"],
                    "B": [10, 1, 0, 1],
                    "C": [8, 0, 1, 0],
                }
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(ans[0], result[0], check_dtype=False)
        pd.testing.assert_frame_equal(ans[1], result[1], check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
result = (result1, result2)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
