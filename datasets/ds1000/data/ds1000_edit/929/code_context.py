import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.linear_model import LogisticRegression


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            GridSearch_fitted = GridSearchCV(LogisticRegression(), {"C": [1, 2, 3]})
            GridSearch_fitted.fit(np.random.randn(50, 4), np.random.randint(0, 2, 50))
        return GridSearch_fitted

    def generate_ans(data):
        def ans1(GridSearch_fitted):
            full_results = pd.DataFrame(GridSearch_fitted.cv_results_).sort_values(
                by="mean_fit_time", ascending=True
            )
            return full_results

        def ans2(GridSearch_fitted):
            full_results = pd.DataFrame(GridSearch_fitted.cv_results_).sort_values(
                by="mean_fit_time", ascending=False
            )
            return full_results

        return ans1(copy.deepcopy(data)), ans2(copy.deepcopy(data))

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans[0], check_dtype=False)
        return 1
    except:
        pass
    try:
        pd.testing.assert_frame_equal(result, ans[1], check_dtype=False)
        return 1
    except:
        pass
    return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
GridSearch_fitted = test_input
[insert]
result = full_results
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
