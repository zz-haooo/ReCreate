import numpy as np
import copy
import scipy.optimize


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            starting_point = [1.8, 1.7]
            direction = [-1, -1]
        elif test_case_id == 2:
            starting_point = [50, 37]
            direction = [-1, -1]
        return starting_point, direction

    def generate_ans(data):
        _a = data

        def test_func(x):
            return (x[0]) ** 2 + (x[1]) ** 2

        def test_grad(x):
            return [2 * x[0], 2 * x[1]]

        starting_point, direction = _a
        result = scipy.optimize.line_search(
            test_func, test_grad, np.array(starting_point), np.array(direction)
        )[0]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert abs(result - ans) <= 1e-5
    return 1


exec_context = r"""
import scipy
import scipy.optimize
import numpy as np
def test_func(x):
    return (x[0])**2+(x[1])**2
def test_grad(x):
    return [2*x[0],2*x[1]]
starting_point, direction = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
