import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A = np.array([1, 2, 3, 4, 5, 6])
            ncol = 2
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.rand(20)
            ncol = 5
        return A, ncol

    def generate_ans(data):
        _a = data
        A, ncol = _a
        B = np.reshape(A, (-1, ncol))
        return B

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, ncol = test_input
[insert]
result = B
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
