import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            return None

    def generate_ans(data):
        none_input = data
        np.random.seed(0)
        r_old = np.random.randint(3, size=(100, 2000)) - 1
        np.random.seed(0)
        r_new = np.random.randint(3, size=(100, 2000)) - 1
        return r_old, r_new

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    r_old, r_new = result
    assert id(r_old) != id(r_new)
    np.testing.assert_array_equal(r_old, r_new)
    return 1


exec_context = r"""
import numpy as np
[insert]
result = [r_old, r_new]
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
    assert "randint" in tokens
