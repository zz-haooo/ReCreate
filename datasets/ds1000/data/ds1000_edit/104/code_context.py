import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        for i in df.index:
            for col in list(df):
                if type(df.loc[i, col]) == str:
                    if "&AMP;" in df.loc[i, col]:
                        df.loc[i, col] = df.loc[i, col].replace("&AMP;", "&")
                        df.loc[i, col] = (
                            df.loc[i, col] + " = " + str(eval(df.loc[i, col]))
                        )
        df.replace("&AMP;", "&", regex=True)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "A": ["1 &AMP; 1", "BB", "CC", "DD", "1 &AMP; 0"],
                    "B": range(5),
                    "C": ["0 &AMP; 0"] * 5,
                }
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
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
