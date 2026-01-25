import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.repeat(np.arange(1, 6).reshape(1, -1), 3, axis=0)
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(3, 4)
        elif test_case_id == 3:
            a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        elif test_case_id == 4:
            a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 3]])
        elif test_case_id == 5:
            a = np.array([[1, 1, 1], [2, 2, 1], [1, 1, 1]])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = np.isclose(a, a[0], atol=0).all()
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
import numpy as np
a = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(5):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens
