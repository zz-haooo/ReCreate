import copy
import tokenize, io
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            sA = sparse.random(10, 10, density=0.1, format="lil", random_state=42)
        return sA

    def generate_ans(data):
        _a = data
        sA = _a
        rows, cols = sA.nonzero()
        sA[cols, rows] = sA[rows, cols]
        return sA

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert len(sparse.find(result != ans)[0]) == 0
    return 1


exec_context = r"""
import numpy as np
from scipy.sparse import lil_matrix
def f(sA):
[insert]
sA = test_input
result = f(sA)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens
