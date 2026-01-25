import numpy as np
import copy
from scipy import stats


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            stddev = 2.0785
            mu = 1.744
            x = 25
        elif test_case_id == 2:
            stddev = 2
            mu = 1
            x = 20
        return x, mu, stddev

    def generate_ans(data):
        _a = data
        x, mu, stddev = _a
        result = stats.lognorm(s=stddev, scale=np.exp(mu)).cdf(x)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy import stats
x, mu, stddev = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
