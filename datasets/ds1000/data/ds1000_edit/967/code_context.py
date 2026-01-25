import torch
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            list_of_tensors = [torch.randn(3), torch.randn(3), torch.randn(3)]
        return list_of_tensors

    def generate_ans(data):
        list_of_tensors = data
        tensor_of_tensors = torch.stack((list_of_tensors))
        return tensor_of_tensors

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
list_of_tensors = test_input
[insert]
result = tensor_of_tensors
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens and "while" not in tokens
