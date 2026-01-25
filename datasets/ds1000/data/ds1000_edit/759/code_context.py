import numpy as np
import copy
from scipy import stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            A = np.random.randn(10)
            B = np.random.randn(10)
        return A, B

    def generate_ans(data):
        _a = data
        pre_course_scores, during_course_scores = _a
        p_value = stats.ranksums(pre_course_scores, during_course_scores).pvalue
        return p_value

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert abs(result - ans) <= 1e-5
    return 1


exec_context = r"""
import numpy as np
from scipy import stats
pre_course_scores, during_course_scores = test_input
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
