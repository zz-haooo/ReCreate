import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            stddev = 2.0785
            mu = 1.744
        return mu, stddev

    def generate_ans(data):
        _a = data
        mu, stddev = _a
        expected_value = np.exp(mu + stddev**2 / 2)
        median = np.exp(mu)
        return [expected_value, median]

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy import stats
mu, stddev = test_input
[insert]
result = [expected_value, median]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
