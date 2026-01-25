import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        s = data
        return s.iloc[np.lexsort([s.index, s.values])]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            s = pd.Series(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0.93],
                index=[
                    "146tf150p",
                    "havent",
                    "home",
                    "okie",
                    "thanx",
                    "er",
                    "anything",
                    "lei",
                    "nite",
                    "yup",
                    "thank",
                    "ok",
                    "where",
                    "beerage",
                    "anytime",
                    "too",
                    "done",
                    "645",
                    "tick",
                    "blank",
                ],
            )
        if test_case_id == 2:
            s = pd.Series(
                [1, 1, 1, 1, 1, 1, 1, 0.86, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0.93],
                index=[
                    "146tf150p",
                    "havent",
                    "homo",
                    "okie",
                    "thanx",
                    "er",
                    "anything",
                    "lei",
                    "nite",
                    "yup",
                    "thank",
                    "ok",
                    "where",
                    "beerage",
                    "anytime",
                    "too",
                    "done",
                    "645",
                    "tick",
                    "blank",
                ],
            )
        return s

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_series_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
s = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
