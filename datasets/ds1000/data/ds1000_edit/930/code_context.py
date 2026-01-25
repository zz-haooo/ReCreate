import copy
import sklearn
from sklearn import datasets
from sklearn.svm import SVC


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            iris = datasets.load_iris()
            X = iris.data[:100, :2]
            y = iris.target[:100]
            model = SVC()
            model.fit(X, y)
            fitted_model = model
        return fitted_model

    def generate_ans(data):
        return None

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    return 1


exec_context = r"""import os
import pandas as pd
import numpy as np
if os.path.exists("sklearn_model"):
    os.remove("sklearn_model")
def creat():
    fitted_model = test_input
    return fitted_model
fitted_model = creat()
[insert]
result = None
assert os.path.exists("sklearn_model") and not os.path.isdir("sklearn_model")
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
