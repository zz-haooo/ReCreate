import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(4)
            df = pd.DataFrame(
                np.repeat([1, 2, 3, 4], 4).reshape(4, -1), columns=["a", "b", "c", "d"]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.randint(0, 10, (4,))
            df = pd.DataFrame(
                np.repeat([1, 2, 3, 4], 4).reshape(4, -1), columns=["a", "b", "c", "d"]
            )
        return a, df

    def generate_ans(data):
        _a = data
        a, df = _a
        df = pd.DataFrame(df.values - a[:, None], df.index, df.columns)
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
a, df = test_input
[insert]
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
