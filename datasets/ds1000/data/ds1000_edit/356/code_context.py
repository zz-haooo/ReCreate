import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            a = np.random.rand(3, 3, 3)
            b = np.arange(3 * 3 * 3).reshape((3, 3, 3))
        return a, b

    def generate_ans(data):
        _a = data
        a, b = _a
        sort_indices = np.argsort(a, axis=0)
        static_indices = np.indices(a.shape)
        c = b[sort_indices, static_indices[1], static_indices[2]]
        return c

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, b = test_input
[insert]
result = c
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
