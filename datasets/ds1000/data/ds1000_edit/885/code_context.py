import numpy as np
import copy
import tokenize, io
from scipy.cluster.hierarchy import linkage, cut_tree


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data_matrix = [[0, 0.8, 0.9], [0.8, 0, 0.2], [0.9, 0.2, 0]]
        elif test_case_id == 2:
            data_matrix = [[0, 0.2, 0.9], [0.2, 0, 0.8], [0.9, 0.8, 0]]
        return data_matrix

    def generate_ans(data):
        simM = data
        Z = linkage(np.array(simM), "ward")
        cluster_labels = (
            cut_tree(Z, n_clusters=2)
            .reshape(
                -1,
            )
            .tolist()
        )
        return cluster_labels

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    ret = 0
    try:
        np.testing.assert_equal(result, ans)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_equal(result, 1 - np.array(ans))
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
import scipy.cluster
simM = test_input
[insert]
result = cluster_labels
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "hierarchy" in tokens
