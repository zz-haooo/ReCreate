import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            A = pd.Series(
                np.random.randn(
                    10,
                )
            )
            a = 2
            b = 3
            c = 4
        elif test_case_id == 2:
            np.random.seed(42)
            A = pd.Series(
                np.random.randn(
                    30,
                )
            )
            a, b, c = np.random.randint(2, 10, (3,))
        return A, a, b, c

    def generate_ans(data):
        _a = data
        A, a, b, c = _a
        B = np.empty(len(A))
        for k in range(0, len(B)):
            if k == 0:
                B[k] = a * A[k]
            elif k == 1:
                B[k] = a * A[k] + b * B[k - 1]
            else:
                B[k] = a * A[k] + b * B[k - 1] + c * B[k - 2]
        return B

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
A, a, b, c = test_input
[insert]
result = B
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
