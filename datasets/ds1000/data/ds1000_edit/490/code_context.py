import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            a = (np.random.rand(5, 50) - 0.5) * 50
            n1 = [1, 2, 3, 4, -5]
            n2 = [6, 7, 8, 9, 10]
        return a, n1, n2

    def generate_ans(data):
        _a = data
        arr, n1, n2 = _a
        for a, t1, t2 in zip(arr, n1, n2):
            temp = a.copy()
            a[np.where(temp < t1)] = 0
            a[np.where(temp >= t2)] = 30
            a[np.logical_and(temp >= t1, temp < t2)] += 5
        return arr

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
arr, n1, n2 = test_input
[insert]
result = arr
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
