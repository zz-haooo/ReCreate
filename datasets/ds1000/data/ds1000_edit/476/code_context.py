import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = {
                "D": [2015, 2015, 2015, 2015, 2016, 2016, 2016, 2017, 2017, 2017],
                "Q": np.arange(10),
            }
            name = "Q_cum"
        elif test_case_id == 2:
            data = {
                "D": [1995, 1995, 1996, 1996, 1997, 1999, 1999, 1999, 2017, 2017],
                "Q": 2 * np.arange(10),
            }
            name = "Q_cum"
        return data, name

    def generate_ans(data):
        _a = data
        data, name = _a
        df = pd.DataFrame(data)
        df[name] = df.groupby("D").cumsum()
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans, check_dtype=False)
    return 1


exec_context = r"""
import pandas as pd
import numpy as np
data, name = test_input
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
