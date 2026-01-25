import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
            y = [[3, 3, 3], [3, 3, 3], [3, 3, 1]]
        elif test_case_id == 2:
            np.random.seed(42)
            dim1 = np.random.randint(5, 10)
            dim2 = np.random.randint(6, 10)
            x = np.random.rand(dim1, dim2)
            y = np.random.rand(dim1, dim2)
        return x, y

    def generate_ans(data):
        _a = data
        x, y = _a
        x_new = np.array(x)
        y_new = np.array(y)
        z = x_new + y_new
        return z

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x, y = test_input
[insert]
result = z
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
    assert "while" not in tokens and "for" not in tokens
