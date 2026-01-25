import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            softmax_output = torch.FloatTensor(
                [[0.2, 0.1, 0.7], [0.6, 0.2, 0.2], [0.1, 0.8, 0.1]]
            )
        elif test_case_id == 2:
            softmax_output = torch.FloatTensor(
                [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
            )
        return softmax_output

    def generate_ans(data):
        softmax_output = data
        y = torch.argmax(softmax_output, dim=1).view(-1, 1)
        return y

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
softmax_output = test_input
[insert]
result = y
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
