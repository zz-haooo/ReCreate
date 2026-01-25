import numpy as np
import copy
import tokenize, io
import scipy.interpolate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            array = np.random.randint(0, 9, size=(10, 10, 10))
            x = np.linspace(0, 10, 10)
            x_new = np.linspace(0, 10, 100)
        return x, array, x_new

    def generate_ans(data):
        _a = data
        x, array, x_new = _a
        new_array = scipy.interpolate.interp1d(x, array, axis=0)(x_new)
        return new_array

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == np.ndarray
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.interpolate
x, array, x_new = test_input
[insert]
result = new_array
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens
