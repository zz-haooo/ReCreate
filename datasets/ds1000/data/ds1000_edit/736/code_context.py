import numpy as np
import copy
import tokenize, io
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.ones((2, 2))
            b = sparse.csr_matrix(a)
        elif test_case_id == 2:
            a = []
            b = sparse.csr_matrix([])
        return a, b

    def generate_ans(data):
        _a = data
        a, b = _a
        b = sparse.csr_matrix(a)
        b.setdiag(0)
        b.eliminate_zeros()
        return b

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == type(ans)
    assert len(sparse.find(result != ans)[0]) == 0
    assert result.nnz == ans.nnz
    return 1


exec_context = r"""
from scipy import sparse
import numpy as np
a, b = test_input
[insert]
result = b
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
    assert (
        "toarray" not in tokens
        and "array" not in tokens
        and "todense" not in tokens
        and "A" not in tokens
    )
