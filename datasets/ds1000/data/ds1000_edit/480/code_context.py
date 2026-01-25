import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array([0, 1, 1, 1, 3, 1, 5, 5, 5])
            y = np.array([0, 2, 3, 4, 2, 4, 3, 4, 5])
            a = 1
            b = 4
        elif test_case_id == 2:
            np.random.seed(42)
            x = np.random.randint(2, 7, (8,))
            y = np.random.randint(2, 7, (8,))
            a = np.random.randint(2, 7)
            b = np.random.randint(2, 7)
        return x, y, a, b

    def generate_ans(data):
        _a = data
        x, y, a, b = _a
        result = ((x == a) & (y == b)).argmax()
        if x[result] != a or y[result] != b:
            result = -1
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x, y, a, b = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
