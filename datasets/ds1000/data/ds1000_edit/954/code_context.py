import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            lens = torch.LongTensor([3, 5, 4])
        elif test_case_id == 2:
            lens = torch.LongTensor([3, 2, 4, 6, 5])
        return lens

    def generate_ans(data):
        lens = data
        max_len = max(lens)
        mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
        mask = mask.type(torch.LongTensor)
        return mask

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        torch.testing.assert_close(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
lens = test_input
def get_mask(lens):
[insert]
mask = get_mask(lens)
result = mask
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
