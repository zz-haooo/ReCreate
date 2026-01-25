import numpy as np
import pandas as pd
import copy
from sklearn import preprocessing


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            data = pd.DataFrame(
                np.random.rand(3, 3),
                index=["first", "second", "third"],
                columns=["c1", "c2", "c3"],
            )
        return data

    def generate_ans(data):
        data = data
        df_out = pd.DataFrame(
            preprocessing.scale(data), index=data.index, columns=data.columns
        )
        return df_out

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False, check_exact=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
data = test_input
[insert]
result = df_out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
