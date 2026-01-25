import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df.index = df.index.set_levels(
            [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            index = pd.MultiIndex.from_tuples(
                [("abc", "3/1/1994"), ("abc", "9/1/1994"), ("abc", "3/1/1995")],
                names=("name", "datetime"),
            )
            df = pd.DataFrame({"fee": [100, 90, 80], "credits": [7, 8, 9]}, index=index)
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
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
