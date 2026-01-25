import numpy as np
import copy
import scipy.ndimage


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.zeros((5, 5))
            a[1:4, 1:4] = np.arange(3 * 3).reshape((3, 3))
        return a

    def generate_ans(data):
        _a = data
        a = _a
        b = scipy.ndimage.median_filter(a, size=(3, 3), origin=(0, 1))
        return b

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.ndimage
a = test_input
[insert]
result = b
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
