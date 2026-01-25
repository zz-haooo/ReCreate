import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            V = sparse.random(10, 10, density=0.05, format="coo", random_state=42)
            x = 100
        elif test_case_id == 2:
            V = sparse.random(15, 15, density=0.05, format="coo", random_state=42)
            x = 5
        return V, x

    def generate_ans(data):
        _a = data
        V, x = _a
        V.data += x
        return V

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert len(sparse.find(ans != result)[0]) == 0
    return 1


exec_context = r"""
from scipy import sparse
V, x = test_input
[insert]
result = V
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
