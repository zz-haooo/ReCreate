import numpy as np
import pandas as pd
import copy
import tokenize, io
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn
from sklearn.datasets import make_classification


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_classification(n_samples=200, n_features=10, random_state=42)
            X = pd.DataFrame(
                X,
                columns=[
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                ],
            )
            y = pd.Series(y)
        return X, y

    def generate_ans(data):
        X, y = data
        clf = ExtraTreesClassifier(random_state=42)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        column_names = list(X.columns[model.get_support()])
        return column_names

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert type(result) == list
        np.testing.assert_equal(ans, result)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
X, y = test_input
clf = ExtraTreesClassifier(random_state=42)
clf = clf.fit(X, y)
[insert]
result = column_names
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
    assert "SelectFromModel" in tokens
