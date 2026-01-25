import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return pd.DataFrame(df.row.str.split(" ", 1).tolist(), columns=["fips", "row"])

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "row": [
                        "114 AAAAAA",
                        "514 ENENEN",
                        "1926 HAHAHA",
                        "0817 O-O,O-O",
                        "998244353 TTTTTT",
                    ]
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "row": [
                        "00000 UNITED STATES",
                        "01000 ALABAMA",
                        "01001 Autauga County, AL",
                        "01003 Baldwin County, AL",
                        "01005 Barbour County, AL",
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
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
