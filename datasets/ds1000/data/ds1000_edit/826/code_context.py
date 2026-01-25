import numpy as np
import copy
import tokenize, io
import sklearn
import sklearn.svm as suppmach
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification, load_iris


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_classification(
                n_samples=100, n_features=2, n_redundant=0, random_state=42
            )
            x_test = X
        elif test_case_id == 2:
            X, y = load_iris(return_X_y=True)
            x_test = X
        return X, y, x_test

    def generate_ans(data):
        def ans1(data):
            X, y, x_test = data
            svmmodel = suppmach.LinearSVC(random_state=42)
            calibrated_svc = CalibratedClassifierCV(svmmodel, cv=5, method="sigmoid")
            calibrated_svc.fit(X, y)
            proba = calibrated_svc.predict_proba(x_test)
            return proba

        def ans2(data):
            X, y, x_test = data
            svmmodel = suppmach.LinearSVC(random_state=42)
            calibrated_svc = CalibratedClassifierCV(svmmodel, cv=5, method="isotonic")
            calibrated_svc.fit(X, y)
            proba = calibrated_svc.predict_proba(x_test)
            return proba

        def ans3(data):
            X, y, x_test = data
            svmmodel = suppmach.LinearSVC(random_state=42)
            calibrated_svc = CalibratedClassifierCV(
                svmmodel, cv=5, method="sigmoid", ensemble=False
            )
            calibrated_svc.fit(X, y)
            proba = calibrated_svc.predict_proba(x_test)
            return proba

        def ans4(data):
            X, y, x_test = data
            svmmodel = suppmach.LinearSVC(random_state=42)
            calibrated_svc = CalibratedClassifierCV(
                svmmodel, cv=5, method="isotonic", ensemble=False
            )
            calibrated_svc.fit(X, y)
            proba = calibrated_svc.predict_proba(x_test)
            return proba

        return (
            ans1(copy.deepcopy(data)),
            ans2(copy.deepcopy(data)),
            ans3(copy.deepcopy(data)),
            ans4(copy.deepcopy(data)),
        )

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
    try:
        np.testing.assert_allclose(result, ans[2])
        ret = 1
    except:
        pass
    try:
        np.testing.assert_allclose(result, ans[3])
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn.svm as suppmach
X, y, x_test = test_input
svmmodel=suppmach.LinearSVC(random_state=42)
[insert]
result = proba
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
    assert "CalibratedClassifierCV" in tokens
