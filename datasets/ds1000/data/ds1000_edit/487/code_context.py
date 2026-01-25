import numpy as np
import copy
from sklearn.preprocessing import minmax_scale


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            arr = np.array([[1.0, 2.0, 3.0], [0.1, 5.1, 100.1], [0.01, 20.1, 1000.1]])
        elif test_case_id == 2:
            np.random.seed(42)
            arr = np.random.rand(3, 5)
        return arr

    def generate_ans(data):
        _a = data
        arr = _a
        result = minmax_scale(arr.T).T
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
arr = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
