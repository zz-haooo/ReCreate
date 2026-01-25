import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(12).reshape(3, 4)
            del_col = np.array([1, 2, 4, 5])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            del_col = np.random.randint(0, 15, (4,))
        return a, del_col

    def generate_ans(data):
        _a = data
        a, del_col = _a
        mask = del_col <= a.shape[1]
        del_col = del_col[mask] - 1
        result = np.delete(a, del_col, axis=1)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, del_col = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
