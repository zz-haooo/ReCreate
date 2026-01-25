import numpy as np
import copy
from scipy import stats


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            x = np.random.normal(0, 1, 1000)
            y = np.random.normal(0, 1, 1000)
        elif test_case_id == 2:
            np.random.seed(42)
            x = np.random.normal(0, 1, 1000)
            y = np.random.normal(1.1, 0.9, 1000)
        alpha = 0.01
        return x, y, alpha

    def generate_ans(data):
        _a = data
        x, y, alpha = _a
        s, p = stats.ks_2samp(x, y)
        result = p <= alpha
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
from scipy import stats
import numpy as np
x, y, alpha = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
