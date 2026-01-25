import numpy as np
import copy
import tokenize, io
import sklearn
from sklearn.preprocessing import PowerTransformer


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = np.array([[1, 2], [3, 2], [4, 5]])
        return data

    def generate_ans(data):
        def ans1(data):
            data = data
            pt = PowerTransformer(method="box-cox")
            box_cox_data = pt.fit_transform(data)
            return box_cox_data

        def ans2(data):
            pt = PowerTransformer(method="box-cox", standardize=False)
            box_cox_data = pt.fit_transform(data)
            return box_cox_data

        return ans1(copy.deepcopy(data)), ans2(copy.deepcopy(data))

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    ret = 0
    try:
        np.testing.assert_allclose(result, ans[0])
        ret = 1
    except:
        pass
    try:
        np.testing.assert_allclose(result, ans[1])
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
data = test_input
[insert]
result = box_cox_data
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
    assert "sklearn" in tokens
