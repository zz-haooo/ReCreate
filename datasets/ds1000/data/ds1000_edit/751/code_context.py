import numpy as np
import itertools
import copy
import scipy.spatial.distance


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            example_array = np.array(
                [
                    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 2, 2, 0, 6, 0, 3, 3, 3],
                    [0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                    [1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 3],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 5, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                ]
            )
        return example_array

    def generate_ans(data):
        _a = data
        example_array = _a
        n = example_array.max() + 1
        indexes = []
        for k in range(1, n):
            tmp = np.nonzero(example_array == k)
            tmp = np.asarray(tmp).T
            indexes.append(tmp)
        result = np.zeros((n - 1, n - 1))
        for i, j in itertools.combinations(range(n - 1), 2):
            d2 = scipy.spatial.distance.cdist(
                indexes[i], indexes[j], metric="sqeuclidean"
            )
            result[i, j] = result[j, i] = d2.min() ** 0.5
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.spatial.distance
example_array = test_input
def f(example_array):
[insert]
result = f(example_array)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
