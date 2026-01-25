import numpy as np
import pandas as pd
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.ones(5)
        elif test_case_id == 2:
            a = np.array([1, 1, 4, 5, 1, 4])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        a_pt = torch.Tensor(a)
        return a_pt

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == torch.Tensor
    torch.testing.assert_close(result, ans, check_dtype=False)
    return 1


exec_context = r"""
import torch
import numpy as np
a = test_input
[insert]
result = a_pt
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
