import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array([[1400, 1500, 1600, np.nan], [1800, np.nan, np.nan, 1700]])
        elif test_case_id == 2:
            x = np.array([[1, 2, np.nan], [3, np.nan, np.nan]])
        elif test_case_id == 3:
            x = np.array([[5, 5, np.nan, np.nan, 2], [3, 4, 5, 6, 7]])
        return x

    def generate_ans(data):
        _a = data
        x = _a
        result = [x[i, row] for i, row in enumerate(~np.isnan(x))]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    for arr1, arr2 in zip(ans, result):
        np.testing.assert_array_equal(arr1, arr2)
    return 1


exec_context = r"""
import numpy as np
x = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
