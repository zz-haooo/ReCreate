import numpy as np
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array(
                [
                    np.array([0.5, 1.0, 2.0], dtype=np.float16),
                    np.array([4.0, 6.0, 8.0], dtype=np.float16),
                ],
                dtype=object,
            )
        elif test_case_id == 2:
            x = np.array(
                [
                    np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float16),
                    np.array([4.0, 6.0, 8.0, 9.0], dtype=np.float16),
                    np.array([4.0, 6.0, 8.0, 9.0], dtype=np.float16),
                ],
                dtype=object,
            )
        return x

    def generate_ans(data):
        x_array = data
        x_tensor = torch.from_numpy(x_array.astype(float))
        return x_tensor

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert torch.is_tensor(result)
        torch.testing.assert_close(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
x_array = test_input
def Convert(a):
[insert]
x_tensor = Convert(x_array)
result = x_tensor
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
