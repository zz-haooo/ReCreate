import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([[1, 2], [3, 4]])
            pos = [1, 2]
            element = np.array([[3, 5], [6, 6]])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(100, 10)
            pos = sorted(np.random.randint(0, 99, (5,)))
            element = np.random.rand(5, 10)
        return a, pos, element

    def generate_ans(data):
        _a = data
        a, pos, element = _a
        pos = np.array(pos) - np.arange(len(element))
        a = np.insert(a, pos, element, axis=0)
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, pos, element = test_input
[insert]
result = a
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
    assert "insert" in tokens
