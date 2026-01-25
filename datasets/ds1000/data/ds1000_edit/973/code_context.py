import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            x = torch.arange(70 * 3 * 2).view(70, 3, 2)
            select_ids = torch.randint(0, 3, size=(70, 1))
            ids = torch.zeros(size=(70, 3))
            for i in range(3):
                ids[i][select_ids[i]] = 1
        return ids, x

    def generate_ans(data):
        ids, x = data
        ids = torch.argmax(ids, 1, True)
        idx = ids.repeat(1, 2).view(70, 1, 2)
        result = torch.gather(x, 1, idx)
        result = result.squeeze(1)
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
ids, x = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
