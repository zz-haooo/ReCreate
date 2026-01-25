import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [-1, 2, 5, 100]
            y = [123, 456, 789, 1255]
            degree = 3
        elif test_case_id == 2:
            np.random.seed(42)
            x = (np.random.rand(100) - 0.5) * 10
            y = (np.random.rand(100) - 0.5) * 10
            degree = np.random.randint(3, 7)
        return x, y, degree

    def generate_ans(data):
        _a = data
        x, y, degree = _a
        result = np.polyfit(x, y, degree)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
x, y, degree = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
