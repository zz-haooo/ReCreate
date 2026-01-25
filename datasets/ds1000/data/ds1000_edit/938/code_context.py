import numpy as np
import pandas as pd
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            x = torch.rand(4, 4)
        elif test_case_id == 2:
            x = torch.rand(6, 6)
        return x

    def generate_ans(data):
        x = data
        px = pd.DataFrame(x.numpy())
        return px

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert type(result) == pd.DataFrame
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
x = test_input
[insert]
result = px
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
