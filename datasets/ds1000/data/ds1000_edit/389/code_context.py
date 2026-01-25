import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array(
                [
                    [[0, 1, 2], [6, 7, 8]],
                    [[3, 4, 5], [9, 10, 11]],
                    [[12, 13, 14], [18, 19, 20]],
                    [[15, 16, 17], [21, 22, 23]],
                ]
            )
            h = 4
            w = 6
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(6, 3, 4)
            h = 6
            w = 12
        return a, h, w

    def generate_ans(data):
        _a = data
        a, h, w = _a
        n, nrows, ncols = a.shape
        result = a.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, h, w = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
