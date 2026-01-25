import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = np.array([4, 2, 5, 6, 7, 5, 4, 3, 5, 7])
            width = 3
        elif test_case_id == 2:
            np.random.seed(42)
            data = np.random.rand(np.random.randint(5, 10))
            width = 4
        return data, width

    def generate_ans(data):
        _a = data
        data, bin_size = _a
        bin_data_max = (
            data[: (data.size // bin_size) * bin_size].reshape(-1, bin_size).max(axis=1)
        )
        return bin_data_max

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans, atol=1e-2)
    return 1


exec_context = r"""
import numpy as np
data, bin_size = test_input
[insert]
result = bin_data_max
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
