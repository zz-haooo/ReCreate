import numpy as np
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            torch.random.manual_seed(42)
            a = torch.randn(1, 3, 10, 40, 1)
        return a

    def generate_ans(data):
        a = data
        Temp = a.unfold(3, 10, 1)
        tensors_31 = []
        for i in range(Temp.shape[3]):
            tensors_31.append(Temp[:, :, :, i, :].view(1, 3, 10, 10, 1).numpy())
        tensors_31 = torch.from_numpy(np.array(tensors_31))
        return tensors_31

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert len(ans) == len(result)
        for i in range(len(ans)):
            torch.testing.assert_close(result[i], ans[i], check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
a = test_input
chunk_dim=10
[insert]
result = tensors_31
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
