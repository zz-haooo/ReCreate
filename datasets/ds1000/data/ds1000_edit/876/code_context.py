import numpy as np
import copy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            f = [["f1", "f2", "f3"], ["f2", "f4", "f5", "f6"], ["f1", "f2"]]
        elif test_case_id == 2:
            f = [
                ["t1"],
                ["t2", "t5", "t7"],
                ["t1", "t2", "t3", "t4", "t5"],
                ["t4", "t5", "t6"],
            ]
        return f

    def generate_ans(data):
        f = data
        new_f = MultiLabelBinarizer().fit_transform(f)
        return new_f

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
f = test_input
[insert]
result = new_f
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
