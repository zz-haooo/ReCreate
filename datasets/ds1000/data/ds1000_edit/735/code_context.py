import numpy as np
import copy
import scipy.spatial
import scipy.optimize


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(100)
            points1 = np.array(
                [(x, y) for x in np.linspace(-1, 1, 7) for y in np.linspace(-1, 1, 7)]
            )
            N = points1.shape[0]
            points2 = 2 * np.random.rand(N, 2) - 1
        return points1, N, points2

    def generate_ans(data):
        _a = data
        points1, N, points2 = _a
        C = scipy.spatial.distance.cdist(points1, points2, metric="minkowski", p=1)
        _, result = scipy.optimize.linear_sum_assignment(C)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.spatial
import scipy.optimize
points1, N, points2 = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
