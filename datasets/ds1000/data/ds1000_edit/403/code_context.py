import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            dims = (3, 4, 2)
            np.random.seed(42)
            a = np.random.rand(*dims)
            index = (1, 0, 1)
        elif test_case_id == 2:
            np.random.seed(42)
            dims = np.random.randint(8, 10, (5,))
            a = np.random.rand(*dims)
            index = np.random.randint(0, 7, (5,))
        return dims, a, index

    def generate_ans(data):
        _a = data
        dims, a, index = _a
        result = np.ravel_multi_index(index, dims=dims, order="C")
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
dims, a, index = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
