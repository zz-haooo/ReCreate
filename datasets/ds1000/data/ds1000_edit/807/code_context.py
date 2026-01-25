import numpy as np
import copy
from scipy.optimize import fsolve


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            xdata = np.arange(4) + 3
            adata = np.random.randint(0, 10, (4,))
        return xdata, adata

    def generate_ans(data):
        _a = data

        def eqn(x, a, b):
            return x + 2 * a - b**2

        xdata, adata = _a
        A = np.array(
            [
                fsolve(lambda b, x, a: eqn(x, a, b), x0=0, args=(x, a))[0]
                for x, a in zip(xdata, adata)
            ]
        )
        temp = -A
        result = np.zeros((len(A), 2))
        result[:, 0] = A
        result[:, 1] = temp
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.optimize import fsolve
def eqn(x, a, b):
    return x + 2*a - b**2
xdata, adata = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
