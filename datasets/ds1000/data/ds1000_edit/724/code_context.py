import numpy as np
import copy
import scipy.interpolate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            points = np.array(
                [
                    [27.827, 18.53, -30.417],
                    [24.002, 17.759, -24.782],
                    [22.145, 13.687, -33.282],
                    [17.627, 18.224, -25.197],
                    [29.018, 18.841, -38.761],
                    [24.834, 20.538, -33.012],
                    [26.232, 22.327, -27.735],
                    [23.017, 23.037, -29.23],
                    [28.761, 21.565, -31.586],
                    [26.263, 23.686, -32.766],
                ]
            )
            values = np.array(
                [0.205, 0.197, 0.204, 0.197, 0.212, 0.208, 0.204, 0.205, 0.211, 0.215]
            )
            request = np.array([[25, 20, -30]])
        return points, values, request

    def generate_ans(data):
        _a = data
        points, V, request = _a
        result = scipy.interpolate.griddata(points, V, request)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans, atol=1e-3)
    return 1


exec_context = r"""
import numpy as np
import scipy.interpolate
points, V, request = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
