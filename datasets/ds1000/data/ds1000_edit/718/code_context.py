import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import scipy.stats
import copy
import io
from scipy import integrate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            z_scores = [-3, -2, 0, 2, 2.5]
            mu = 3
            sigma = 4
        return z_scores, mu, sigma

    def generate_ans(data):
        _a = data
        z_scores, mu, sigma = _a
        temp = np.array(z_scores)
        p_values = scipy.stats.norm.cdf(temp)
        return p_values

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.stats
z_scores, mu, sigma = test_input
[insert]
result = p_values
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
