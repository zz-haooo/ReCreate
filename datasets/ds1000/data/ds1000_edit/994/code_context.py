import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = torch.Tensor([[1, 2, 3], [1, 2, 3]])
            b = torch.Tensor([[5, 6, 7], [5, 6, 7]])
        elif test_case_id == 2:
            a = torch.Tensor([[3, 2, 1], [1, 2, 3]])
            b = torch.Tensor([[7, 6, 5], [5, 6, 7]])
        elif test_case_id == 3:
            a = torch.Tensor([[3, 2, 1, 1, 2], [1, 1, 1, 2, 3], [9, 9, 5, 6, 7]])
            b = torch.Tensor([[1, 4, 7, 6, 5], [9, 9, 5, 6, 7], [9, 9, 5, 6, 7]])
        return a, b

    def generate_ans(data):
        a, b = data
        c = (a[:, -1:] + b[:, :1]) / 2
        result = torch.cat((a[:, :-1], c, b[:, 1:]), dim=1)
        return result

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
a, b = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
