import numpy as np
import copy
import scipy.stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            a = np.random.randn(40)
            b = 4 * np.random.randn(50)
        return a, b

    def generate_ans(data):
        _a = data
        a, b = _a
        _, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)
        return p_value

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert abs(ans - result) <= 1e-5
    return 1


exec_context = r"""
import numpy as np
import scipy.stats
a, b = test_input
[insert]
result = p_value
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
