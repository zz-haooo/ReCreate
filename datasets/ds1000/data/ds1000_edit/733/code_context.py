import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            c1 = sparse.csr_matrix([[0, 0, 1, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
            c2 = sparse.csr_matrix([[0, 3, 4, 0], [0, 0, 0, 5], [6, 7, 0, 8]])
        return c1, c2

    def generate_ans(data):
        _a = data
        c1, c2 = _a
        Feature = sparse.vstack((c1, c2))
        return Feature

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == sparse.csr_matrix
    assert len(sparse.find(ans != result)[0]) == 0
    return 1


exec_context = r"""
from scipy import sparse
c1, c2 = test_input
[insert]
result = Feature
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
