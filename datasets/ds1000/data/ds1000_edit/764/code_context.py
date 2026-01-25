import numpy as np
import copy
import scipy.interpolate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            s = np.linspace(-1, 1, 50)
            t = np.linspace(-2, 0, 50)
        return s, t

    def generate_ans(data):
        _a = data
        s, t = _a
        x, y = np.ogrid[-1:1:10j, -2:0:10j]
        z = (x + y) * np.exp(-6.0 * (x * x + y * y))
        spl = scipy.interpolate.RectBivariateSpline(x, y, z)
        result = spl(s, t, grid=False)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.interpolate
s, t = test_input
def f(s, t):
    x, y = np.ogrid[-1:1:10j,-2:0:10j]
    z = (x + y)*np.exp(-6.0 * (x * x + y * y))
[insert]
result = f(s, t)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
