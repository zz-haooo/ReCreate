import numpy as np
import copy
from scipy.stats import rankdata


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = [1, 2, 3, 4, 3, 2, 3, 4]
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(26, 30))
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = len(a) - rankdata(a).astype(int)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.stats import rankdata
a = test_input
def f(a):
[insert]
result = f(a)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
