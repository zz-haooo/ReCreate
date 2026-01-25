import numpy as np
import math
import copy
import scipy.integrate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = 2.5
            u = 1
            o2 = 3
        elif test_case_id == 2:
            x = -2.5
            u = 2
            o2 = 4
        return x, u, o2

    def generate_ans(data):
        _a = data

        def NDfx(x):
            return (1 / math.sqrt((2 * math.pi))) * (math.e ** ((-0.5) * (x**2)))

        x, u, o2 = _a
        norm = (x - u) / o2
        prob = scipy.integrate.quad(NDfx, -np.inf, norm)[0]
        return prob

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import scipy.integrate
import math
import numpy as np
x, u, o2 = test_input
def NDfx(x):
    return((1/math.sqrt((2*math.pi)))*(math.e**((-.5)*(x**2))))
def f(x, u, o2):
[insert]
result = f(x, u, o2)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
