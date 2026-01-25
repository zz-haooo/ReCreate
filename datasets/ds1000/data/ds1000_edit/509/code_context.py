import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            im = np.array(
                [
                    [1, 1, 1, 1, 1, 5],
                    [1, 0, 0, 1, 2, 0],
                    [2, 1, 0, 0, 1, 0],
                    [1, 0, 0, 7, 1, 0],
                    [1, 0, 0, 0, 0, 0],
                ]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            im = np.random.randint(0, 10, (10, 12))
            im[:, 0] = 5
            im[-1, :] = 5
        elif test_case_id == 3:
            im = np.ones((10, 10))
        return im

    def generate_ans(data):
        _a = data
        im = _a
        mask = im == 0
        rows = np.flatnonzero((mask).sum(axis=1))
        cols = np.flatnonzero((mask).sum(axis=0))
        if rows.shape[0] == 0:
            result = np.array([])
        else:
            result = im[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
im = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens
