import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        cols = list(df.filter(like="col"))
        df["index_original"] = df.groupby(cols)[cols[0]].transform("idxmax")
        for i in range(len(df)):
            i = len(df) - 1 - i
            origin = df.loc[i, "index_original"]
            if i <= origin:
                continue
            if origin == df.loc[origin, "index_original"]:
                df.loc[origin, "index_original"] = i
            df.loc[i, "index_original"] = df.loc[origin, "index_original"]
        return df[df.duplicated(subset=cols, keep="last")]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                data=[
                    [1, 1, 2, 5],
                    [1, 3, 4, 1],
                    [4, 1, 2, 5],
                    [5, 1, 4, 9],
                    [1, 1, 2, 5],
                ],
                columns=["val", "col1", "col2", "3col"],
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                data=[
                    [4, 1, 2, 5],
                    [1, 1, 2, 5],
                    [1, 3, 4, 1],
                    [9, 9, 1, 4],
                    [1, 1, 2, 5],
                ],
                columns=["val", "col1", "col2", "3col"],
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
