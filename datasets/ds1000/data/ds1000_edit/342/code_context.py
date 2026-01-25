import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            X = np.random.randint(2, 10, (5, 6))
        return X

    def generate_ans(data):
        _a = data
        X = _a
        result = []
        for value in X.flat:
            result.append(value)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(sorted(result), sorted(ans))
    return 1


exec_context = r"""
import numpy as np
X = test_input
def f(X):
[insert]
result = f(X)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
