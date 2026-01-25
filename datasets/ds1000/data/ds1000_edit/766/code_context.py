import numpy as np
import copy
import scipy.spatial


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            points = [[0, 0], [1, 4], [2, 3], [4, 1], [1, 1], [2, 2], [5, 3]]
            extraPoints = [[0.5, 0.2], [3, 0], [4, 0], [5, 0], [4, 3]]
        elif test_case_id == 2:
            np.random.seed(42)
            points = (np.random.rand(15, 2) - 0.5) * 100
            extraPoints = (np.random.rand(10, 2) - 0.5) * 100
        return points, extraPoints

    def generate_ans(data):
        _a = data
        points, extraPoints = _a
        vor = scipy.spatial.Voronoi(points)
        kdtree = scipy.spatial.cKDTree(points)
        _, index = kdtree.query(extraPoints)
        result = vor.point_region[index]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import scipy.spatial
points, extraPoints = test_input
vor = scipy.spatial.Voronoi(points)
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
