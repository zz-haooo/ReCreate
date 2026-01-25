import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            lat = np.array([[10, 20, 30], [20, 11, 33], [21, 20, 10]])
            lon = np.array([[100, 102, 103], [105, 101, 102], [100, 102, 103]])
            val = np.array([[17, 2, 11], [86, 84, 1], [9, 5, 10]])
        elif test_case_id == 2:
            np.random.seed(42)
            lat = np.random.rand(5, 6)
            lon = np.random.rand(5, 6)
            val = np.random.rand(5, 6)
        return lat, lon, val

    def generate_ans(data):
        _a = data
        lat, lon, val = _a
        df = pd.DataFrame({"lat": lat.ravel(), "lon": lon.ravel(), "val": val.ravel()})
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans, check_dtype=False)
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
lat, lon, val = test_input
def f(lat, lon,val):
[insert]
result = f(lat, lon, val)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
