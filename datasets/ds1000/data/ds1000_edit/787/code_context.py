import numpy as np
import copy
import scipy.optimize


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            a = np.random.rand(3, 5)
            x_true = np.array([10, 13, 5, 8, 40])
            y = a.dot(x_true**2)
            x0 = np.array([2, 3, 1, 4, 20])
            x_bounds = x_true / 2
        return a, x_true, y, x0, x_bounds

    def generate_ans(data):
        _a = data
        a, x_true, y, x0, x_lower_bounds = _a

        def residual_ans(x, a, y):
            s = ((y - a.dot(x**2)) ** 2).sum()
            return s

        bounds = [[x, None] for x in x_lower_bounds]
        out = scipy.optimize.minimize(
            residual_ans, x0=x0, args=(a, y), method="L-BFGS-B", bounds=bounds
        ).x
        return out

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import scipy.optimize
import numpy as np
a, x_true, y, x0, x_lower_bounds = test_input
[insert]
result = out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
