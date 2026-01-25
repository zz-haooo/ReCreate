import numpy as np
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            corpus = [
                "This is the first document.",
                "This document is the second document.",
                "And this is the first piece of news",
                "Is this the first document? No, it'the fourth document",
                "This is the second news",
            ]
            y = [0, 0, 1, 0, 1]
        return corpus, y

    def generate_ans(data):
        corpus, y = data
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        svc = LinearSVC(penalty="l1", dual=False)
        svc.fit(X, y)
        selected_feature_names = np.asarray(vectorizer.get_feature_names_out())[
            np.flatnonzero(svc.coef_)
        ]
        return selected_feature_names

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
corpus, y = test_input
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
def solve(corpus, y, vectorizer, X):
[insert]
selected_feature_names = solve(corpus, y, vectorizer, X)
result = selected_feature_names
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
