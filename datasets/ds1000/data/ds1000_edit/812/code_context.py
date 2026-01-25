import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = pd.DataFrame({"A1": [0, 1, 2, 3, 2, 1, 6, 0, 1, 1, 7, 10]})
        return a

    def generate_ans(data):
        _a = data
        a = _a
        weights = (a.values / a.values.sum()).squeeze()
        return weights

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans, atol=1e-2)
    return 1


exec_context = r"""
import scipy.optimize as sciopt
import numpy as np
import pandas as pd
a = test_input
[insert]
result = weights
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
