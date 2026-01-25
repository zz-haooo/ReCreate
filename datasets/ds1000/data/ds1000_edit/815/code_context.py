import numpy as np
import copy
from scipy import signal


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            arr = np.array(
                [
                    [
                        -624.59309896,
                        -624.59309896,
                        -624.59309896,
                        -625.0,
                        -625.0,
                        -625.0,
                    ],
                    [3, 0, 0, 1, 2, 4],
                ]
            )
            n = 2
        elif test_case_id == 2:
            np.random.seed(42)
            arr = (np.random.rand(50, 10) - 0.5) * 10
            n = np.random.randint(2, 4)
        return arr, n

    def generate_ans(data):
        _a = data
        arr, n = _a
        res = signal.argrelextrema(arr, np.less_equal, order=n, axis=1)
        result = np.zeros((res[0].shape[0], 2)).astype(int)
        result[:, 0] = res[0]
        result[:, 1] = res[1]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    assert np.array(result).dtype == np.int64 or np.array(result).dtype == np.int32
    return 1


exec_context = r"""
import numpy as np
from scipy import signal
arr, n = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
