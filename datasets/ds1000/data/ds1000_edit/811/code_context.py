import numpy as np
import copy
import tokenize, io
import scipy.interpolate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [(2, 2), (1, 2), (2, 3), (3, 2), (2, 1)]
            y = [5, 7, 8, 10, 3]
            eval = [(2.7, 2.3)]
        return x, y, eval

    def generate_ans(data):
        _a = data
        x, y, eval = _a
        result = scipy.interpolate.griddata(x, y, eval)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import scipy.interpolate
x, y, eval = test_input
[insert]
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
    assert "griddata" in tokens
