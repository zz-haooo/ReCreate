import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 8])
            B = np.array([1, 4, 8])
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.randint(0, 10, (20,))
            B = np.array([2, 2, 2])
        elif test_case_id == 3:
            np.random.seed(44)
            A = np.random.randint(0, 10, (20,))
            B = np.array([2, 3, 5])
        return A, B

    def generate_ans(data):
        _a = data
        A, B = _a
        C = A[np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])]
        return C

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, B = test_input
[insert]
result = C
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens
