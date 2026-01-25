import numpy as np
import copy
import scipy.sparse as sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            max_vector_size = 1000
            vectors = [
                np.random.randint(100, size=900),
                np.random.randint(100, size=max_vector_size),
                np.random.randint(100, size=950),
            ]
        elif test_case_id == 2:
            np.random.seed(42)
            max_vector_size = 300
            vectors = [
                np.random.randint(200, size=200),
                np.random.randint(200, size=max_vector_size),
                np.random.randint(200, size=200),
                np.random.randint(200, size=max_vector_size),
                np.random.randint(200, size=200),
            ]
        return vectors, max_vector_size

    def generate_ans(data):
        _a = data
        vectors, max_vector_size = _a
        result = sparse.lil_matrix((len(vectors), max_vector_size))
        for i, v in enumerate(vectors):
            result[i, : v.size] = v
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert len(sparse.find(result != ans)[0]) == 0
    return 1


exec_context = r"""
import numpy as np
import scipy.sparse as sparse
vectors, max_vector_size = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
