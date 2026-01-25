import numpy as np
import torch
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            t = torch.tensor([[-0.2, 0.3], [-0.5, 0.1], [-0.4, 0.2]])
            idx = np.array([1, 0, 1], dtype=np.int32)
        elif test_case_id == 2:
            t = torch.tensor([[-0.2, 0.3], [-0.5, 0.1], [-0.4, 0.2]])
            idx = np.array([1, 1, 0], dtype=np.int32)
        return t, idx

    def generate_ans(data):
        t, idx = data
        idx = 1 - idx
        idxs = torch.from_numpy(idx).long().unsqueeze(1)
        result = t.gather(1, idxs).squeeze(1)
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
t, idx = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens and "while" not in tokens
