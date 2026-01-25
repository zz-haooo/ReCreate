import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        dict, df = data
        df["Date"] = df["Member"].apply(lambda x: dict.get(x)).fillna(np.NAN)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            dict = {"abc": "1/2/2003", "def": "1/5/2017", "ghi": "4/10/2013"}
            df = pd.DataFrame(
                {
                    "Member": ["xyz", "uvw", "abc", "def", "ghi"],
                    "Group": ["A", "B", "A", "B", "B"],
                    "Date": [np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            )
        if test_case_id == 2:
            dict = {"abc": "1/2/2013", "def": "1/5/2027", "ghi": "4/10/2023"}
            df = pd.DataFrame(
                {
                    "Member": ["xyz", "uvw", "abc", "def", "ghi"],
                    "Group": ["A", "B", "A", "B", "B"],
                    "Date": [np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            )
        return dict, df

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
dict, df = test_input
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
