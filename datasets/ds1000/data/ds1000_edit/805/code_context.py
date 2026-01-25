import numpy as np
import copy
import scipy.spatial


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            centroids = np.random.rand(5, 3)
            data = np.random.rand(100, 3)
            k = 3
        return centroids, data, k

    def generate_ans(data):
        _a = data
        centroids, data, k = _a

        def find_k_closest(centroids, data, k=1, distance_norm=2):
            kdtree = scipy.spatial.cKDTree(data)
            distances, indices = kdtree.query(centroids, k, p=distance_norm)
            if k > 1:
                indices = indices[:, -1]
            values = data[indices]
            return indices, values

        result, _ = find_k_closest(centroids, data, k)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import scipy.spatial
import numpy as np
centroids, data, k = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
