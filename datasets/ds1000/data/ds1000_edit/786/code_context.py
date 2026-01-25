import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            a = np.random.rand(3, 5)
            x_true = np.array([10, 13, 5, 8, 40])
            y = a.dot(x_true**2)
            x0 = np.array([2, 3, 1, 4, 20])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(4, 6)
            x_true = np.array([-3, 2, 7, 18, 4, -1])
            y = a.dot(x_true**2)
            x0 = np.array([2, 3, 1, 4, 2, 0])
        return a, y, x0

    def generate_ans(data):
        _a = data
        a, y, x0 = _a
        return a, y

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    a, y = ans

    def residual(x, a, y):
        s = ((y - a.dot(x**2)) ** 2).sum()
        return s

    assert residual(result, a, y) < 1e-5
    return 1


exec_context = r"""
import scipy.optimize
import numpy as np
a, y, x0 = test_input
[insert]
result = out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
