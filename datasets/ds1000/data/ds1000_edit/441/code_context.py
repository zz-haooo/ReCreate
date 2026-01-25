import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([9, 2, 7, 0])
            number = 0
        elif test_case_id == 2:
            a = np.array([1, 2, 3, 5])
            number = 4
        elif test_case_id == 3:
            a = np.array([1, 1, 1, 1])
            number = 1
        return a, number

    def generate_ans(data):
        _a = data
        a, number = _a
        is_contained = number in a
        return is_contained

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
import numpy as np
a, number = test_input
[insert]
result = is_contained
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
