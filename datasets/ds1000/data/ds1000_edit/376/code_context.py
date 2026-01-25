import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            one_ratio = 0.9
            size = 1000
        elif test_case_id == 2:
            size = 100
            one_ratio = 0.8
        return size, one_ratio

    def generate_ans(data):
        _a = data
        return _a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    size, one_ratio = ans
    assert result.shape == (size,)
    assert abs(len(np.where(result == 1)[0]) - size * one_ratio) / size <= 0.05
    assert abs(len(np.where(result == 0)[0]) - size * (1 - one_ratio)) / size <= 0.05
    return 1


exec_context = r"""
import numpy as np
size, one_ratio = test_input
[insert]
result = nums
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
    assert "random" in tokens
