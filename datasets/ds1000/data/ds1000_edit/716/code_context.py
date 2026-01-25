import numpy as np
import copy
from scipy import optimize


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = [-1, 0, -3]
        return a

    def generate_ans(data):
        _a = data
        initial_guess = _a

        def g(params):
            a, b, c = params
            return (
                ((a + b - c) - 2) ** 2
                + ((3 * a - b - c)) ** 2
                + np.sin(b)
                + np.cos(b)
                + 4
            )

        res = optimize.minimize(g, initial_guess)
        result = res.x
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    def g(params):
        a, b, c = params
        return (
            ((a + b - c) - 2) ** 2 + ((3 * a - b - c)) ** 2 + np.sin(b) + np.cos(b) + 4
        )

    assert abs(g(result) - g(ans)) < 1e-2
    return 1


exec_context = r"""
import scipy.optimize as optimize
from math import sqrt, sin, pi, cos
initial_guess = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
