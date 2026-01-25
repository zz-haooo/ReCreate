import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        element = 5
        if test_case_id == 1:
            a = np.ones((41, 12))
            shape = (93, 13)
        elif test_case_id == 2:
            a = np.ones((41, 13))
            shape = (93, 13)
        elif test_case_id == 3:
            a = np.ones((93, 11))
            shape = (93, 13)
        elif test_case_id == 4:
            a = np.ones((42, 10))
            shape = (93, 13)
        return a, shape, element

    def generate_ans(data):
        _a = data
        a, shape, element = _a
        result = np.pad(
            a,
            ((0, shape[0] - a.shape[0]), (0, shape[1] - a.shape[1])),
            "constant",
            constant_values=element,
        )
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, shape, element = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(4):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
