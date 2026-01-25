import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.matrix([[3, 4, 3, 1], [1, 3, 2, 6], [2, 4, 1, 5], [3, 3, 5, 2]])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        U, i, V = np.linalg.svd(a, full_matrices=True)
        i = np.diag(i)
        return i

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
U, i, V = np.linalg.svd(a,full_matrices=True)
[insert]
result = i
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
