import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array([1400, 1500, 1600, np.nan, np.nan, np.nan, 1700])
        elif test_case_id == 2:
            np.random.seed(42)
            x = np.random.rand(20)
            x[np.random.randint(0, 20, 3)] = np.nan
        return x

    def generate_ans(data):
        _a = data
        x = _a
        x = x[~np.isnan(x)]
        return x

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x = test_input
[insert]
result = x
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
