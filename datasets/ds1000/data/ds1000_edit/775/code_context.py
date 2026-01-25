import numpy as np
import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            matrix = np.array(
                [
                    [3.5, 13.0, 28.5, 50.0, 77.5],
                    [-5.0, -23.0, -53.0, -95.0, -149.0],
                    [2.5, 11.0, 25.5, 46.0, 72.5],
                ]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            matrix = np.random.rand(3, 5)
        return matrix

    def generate_ans(data):
        _a = data
        matrix = _a
        result = sparse.spdiags(matrix, (1, 0, -1), 5, 5).T.A
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
from scipy import sparse
import numpy as np
matrix = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
