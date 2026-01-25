import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A = np.arange(16).reshape(4, 4)
            n = 5
        elif test_case_id == 2:
            np.random.seed(42)
            dim = np.random.randint(10, 15)
            A = np.random.rand(dim, dim)
            n = np.random.randint(3, 8)
        return A, n

    def generate_ans(data):
        _a = data
        A, n = _a
        result = np.linalg.matrix_power(A, n)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == np.ndarray
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, n = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "matrix" not in tokens
