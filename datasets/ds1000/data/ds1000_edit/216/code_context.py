import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        Date = list(df.index)
        Date = sorted(Date)
        half = len(list(Date)) // 2
        return max(Date, key=lambda v: Date.count(v)), Date[half]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {"value": [10000, 2000, 2000, 200, 5, 70, 200, 5, 25, 0.02, 12, 0.022]},
                index=[
                    "2014-03-13",
                    "2014-03-21",
                    "2014-03-27",
                    "2014-03-17",
                    "2014-03-17",
                    "2014-03-17",
                    "2014-03-21",
                    "2014-03-27",
                    "2014-03-27",
                    "2014-03-31",
                    "2014-03-31",
                    "2014-03-31",
                ],
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {"value": [10000, 2000, 2000, 200, 5, 70, 200, 5, 25, 0.02, 12, 0.022]},
                index=[
                    "2015-03-13",
                    "2015-03-21",
                    "2015-03-27",
                    "2015-03-17",
                    "2015-03-17",
                    "2015-03-17",
                    "2015-03-21",
                    "2015-03-27",
                    "2015-03-27",
                    "2015-03-31",
                    "2015-03-31",
                    "2015-03-31",
                ],
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert result[0] == ans[0]
        assert result[1] == ans[1]
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
result = (mode_result, median_result)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
