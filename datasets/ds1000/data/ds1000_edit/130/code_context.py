import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["index_original"] = df.groupby(["col1", "col2"]).col1.transform("idxmin")
        return df[df.duplicated(subset=["col1", "col2"], keep="first")]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                data=[[1, 2], [3, 4], [1, 2], [1, 4], [1, 2]], columns=["col1", "col2"]
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                data=[[1, 1], [3, 1], [1, 4], [1, 9], [1, 6]], columns=["col1", "col2"]
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
