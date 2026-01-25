import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            numerator = 98
            denominator = 42
        elif test_case_id == 2:
            np.random.seed(42)
            numerator = np.random.randint(2, 10)
            denominator = np.random.randint(2, 10)
        elif test_case_id == 3:
            numerator = -5
            denominator = 10
        return numerator, denominator

    def generate_ans(data):
        _a = data
        numerator, denominator = _a
        gcd = np.gcd(numerator, denominator)
        result = (numerator // gcd, denominator // gcd)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result[0] == ans[0] and result[1] == ans[1]
    return 1


exec_context = r"""
import numpy as np
numerator, denominator = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
