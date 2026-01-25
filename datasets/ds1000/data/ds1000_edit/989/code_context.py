import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            torch.random.manual_seed(42)
            mask = torch.tensor([[0, 1, 0]]).to(torch.int32)
            clean_input_spectrogram = torch.rand((1, 3, 2))
            output = torch.rand((1, 3, 2))
        return mask, clean_input_spectrogram, output

    def generate_ans(data):
        mask, clean_input_spectrogram, output = data
        for i in range(len(mask[0])):
            if mask[0][i] == 1:
                mask[0][i] = 0
            else:
                mask[0][i] = 1
        output[:, mask[0].to(torch.bool), :] = clean_input_spectrogram[
            :, mask[0].to(torch.bool), :
        ]
        return output

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
mask, clean_input_spectrogram, output = test_input
[insert]
result = output
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
