import numpy as np
import copy
import scipy.optimize


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            y = np.array([1, 7, 20, 50, 79])
            x = np.array([10, 19, 30, 35, 51])
            p0 = (4, 0.1, 1)
        return x, y, p0

    def generate_ans(data):
        _a = data
        x, y, p0 = _a
        result = scipy.optimize.curve_fit(
            lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=p0
        )[0]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.optimize
x, y, p0 = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
