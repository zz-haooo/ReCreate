import numpy as np
import copy
import tokenize, io
from scipy.sparse import csr_matrix


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        elif test_case_id == 2:
            np.random.seed(42)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        elif test_case_id == 3:
            np.random.seed(80)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        elif test_case_id == 4:
            np.random.seed(100)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        return col

    def generate_ans(data):
        _a = data
        col = _a
        mean = col.mean()
        N = col.shape[0]
        sqr = col.copy()  # take a copy of the col
        sqr.data **= 2  # square the data, i.e. just the non-zero data
        standard_deviation = np.sqrt(sqr.sum() / N - col.mean() ** 2)
        return [mean, standard_deviation]

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.sparse import csr_matrix
col = test_input
[insert]
result = [mean, standard_deviation]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(4):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert (
        "toarray" not in tokens
        and "array" not in tokens
        and "todense" not in tokens
        and "A" not in tokens
    )
