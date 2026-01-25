import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A = ["33.33", "33.33", "33.33", "33.37"]
            NA = np.asarray(A)
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.rand(5)
            NA = A.astype(str)
        return A, NA

    def generate_ans(data):
        _a = data
        A, NA = _a
        AVG = np.mean(NA.astype(float), axis=0)
        return AVG

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, NA = test_input
[insert]
result = AVG
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
