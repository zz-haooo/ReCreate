import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
            size = (3, 3)
        return a, size

    def generate_ans(data):
        _a = data
        a, size = _a

        def window(arr, shape=(3, 3)):
            ans = []
            r_win = np.floor(shape[0] / 2).astype(int)
            c_win = np.floor(shape[1] / 2).astype(int)
            x, y = arr.shape
            for j in range(y):
                ymin = max(0, j - c_win)
                ymax = min(y, j + c_win + 1)
                for i in range(x):
                    xmin = max(0, i - r_win)
                    xmax = min(x, i + r_win + 1)
                    ans.append(arr[xmin:xmax, ymin:ymax])
            return ans

        result = window(a, size)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    for arr1, arr2 in zip(ans, result):
        np.testing.assert_allclose(arr1, arr2)
    return 1


exec_context = r"""
import numpy as np
a, size = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
