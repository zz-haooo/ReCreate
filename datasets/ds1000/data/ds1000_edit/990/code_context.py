import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            torch.random.manual_seed(42)
            x = torch.randint(-10, 10, (5,))
            y = torch.randint(-20, 20, (5,))
        return x, y

    def generate_ans(data):
        x, y = data
        mins = torch.min(torch.abs(x), torch.abs(y))
        xSigns = (mins == torch.abs(x)) * torch.sign(x)
        ySigns = (mins == torch.abs(y)) * torch.sign(y)
        finalSigns = xSigns.int() | ySigns.int()
        signed_min = mins * finalSigns
        return signed_min

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
x, y = test_input
[insert]
result = signed_min
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
