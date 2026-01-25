import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(4).reshape(2, 2)
            power = 5
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            power = np.random.randint(6, 10)
        return a, power

    def generate_ans(data):
        _a = data
        a, power = _a
        result = a**power
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, power = test_input
def f(a, power):
[insert]
result = f(a, power)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
