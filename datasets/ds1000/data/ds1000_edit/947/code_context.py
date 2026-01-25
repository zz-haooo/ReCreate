import torch
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            idx = torch.LongTensor([1, 2])
            B = torch.LongTensor([[2, 1, 3], [5, 4, 6]])
        elif test_case_id == 2:
            idx = torch.LongTensor([0, 1, 3])
            B = torch.LongTensor([[1, 2, 3, 777], [4, 999, 5, 6]])
        return idx, B

    def generate_ans(data):
        idx, B = data
        C = B.index_select(1, idx)
        return C

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        torch.testing.assert_close(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
idx, B = test_input
[insert]
result = C
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
    assert "index_select" in tokens
