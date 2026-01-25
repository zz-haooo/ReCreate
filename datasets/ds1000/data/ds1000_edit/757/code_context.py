import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            sa = sparse.random(10, 10, density=0.01, format="lil", random_state=42)
        elif test_case_id == 2:
            sa = sparse.lil_matrix([])
        return sa

    def generate_ans(data):
        _a = data
        sa = _a
        result = sa.count_nonzero() == 0
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
from scipy import sparse
sa = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
