import numpy as np
import copy
from scipy import interpolate as intp


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(0, 4, 1).reshape(2, 2)
            a = a.repeat(2, axis=0).repeat(2, axis=1)
            x_new = np.linspace(0, 2, 4)
            y_new = np.linspace(0, 2, 4)
        return a, x_new, y_new

    def generate_ans(data):
        _a = data
        a, x_new, y_new = _a
        x = np.arange(4)
        y = np.arange(4)
        f = intp.interp2d(x, y, a)
        result = f(x_new, y_new)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy import interpolate as intp
a, x_new, y_new = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
