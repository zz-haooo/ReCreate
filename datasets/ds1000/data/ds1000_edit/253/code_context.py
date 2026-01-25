import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        for i in df.index:
            df.loc[i, "codes"] = sorted(df.loc[i, "codes"])
        df = df.codes.apply(pd.Series)
        cols = list(df)
        for i in range(len(cols)):
            cols[i] += 1
        df.columns = cols
        return df.add_prefix("code_")

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "codes": [
                        [71020],
                        [77085],
                        [36415],
                        [99213, 99287],
                        [99234, 99233, 99234],
                    ]
                }
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "codes": [
                        [71020, 71011],
                        [77085],
                        [99999, 36415],
                        [99213, 99287],
                        [99233, 99232, 99234],
                    ]
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
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
