import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([1.5, -0.4, 1.3])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(20)
        elif test_case_id == 3:
            a = np.array([1.5, -0.4, 1.3, 1.5, 1.3])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        vals, idx = np.unique(a, return_inverse=True)
        b = np.zeros((a.size, vals.size))
        b[np.arange(a.size), idx] = 1
        return b

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
[insert]
result = b
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
