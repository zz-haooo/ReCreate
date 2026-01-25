import numpy as np
import copy
from sklearn.preprocessing import StandardScaler


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = [[1, 1], [2, 3], [3, 2], [1, 1]]
        return data

    def generate_ans(data):
        data = data
        scaler = StandardScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)
        inversed = scaler.inverse_transform(scaled)
        return inversed

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data = test_input
scaler = StandardScaler()
scaler.fit(data)
scaled = scaler.transform(data)
def solve(data, scaler, scaled):
[insert]
inversed = solve(data, scaler, scaled)
result = inversed
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
