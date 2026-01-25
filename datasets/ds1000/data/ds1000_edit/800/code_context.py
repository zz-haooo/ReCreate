import numpy as np
import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            sa = sparse.random(10, 10, density=0.3, format="csr", random_state=42)
        return sa

    def generate_ans(data):
        _a = data
        sa = _a
        sa = sparse.csr_matrix(
            sa.toarray() / np.sqrt(np.sum(sa.toarray() ** 2, axis=0))
        )
        return sa

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == sparse.csr.csr_matrix
    assert len(sparse.find(result != ans)[0]) == 0
    return 1


exec_context = r"""
from scipy import sparse
import numpy as np
import math
sa = test_input
[insert]
result = sa
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
