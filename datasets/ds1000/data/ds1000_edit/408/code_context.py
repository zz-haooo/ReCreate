import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(1, 11)
            index = np.array([0, 1, 0, 0, 0, -1, -1, 2, 2, 1])
        elif test_case_id == 2:
            np.random.seed(42)
            index = np.random.randint(-2, 5, (100,))
            a = np.random.randint(-100, 100, (100,))
        return a, index

    def generate_ans(data):
        _a = data
        a, index = _a
        add = np.max(index)
        mask = index < 0
        index[mask] += add + 1
        uni = np.unique(index)
        result = np.zeros(np.amax(index) + 1)
        for i in uni:
            result[i] = np.min(a[index == i])
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, index = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
