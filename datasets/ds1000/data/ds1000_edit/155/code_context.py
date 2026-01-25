import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        uniq_indx = (
            df.sort_values(by="bank", na_position="last")
            .dropna(subset=["firstname", "lastname", "email"])
            .applymap(lambda s: s.lower() if type(s) == str else s)
            .applymap(lambda x: x.replace(" ", "") if type(x) == str else x)
            .drop_duplicates(subset=["firstname", "lastname", "email"], keep="first")
        ).index
        return df.loc[uniq_indx]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "firstname": ["foo Bar", "Bar Bar", "Foo Bar"],
                    "lastname": ["Foo Bar", "Bar", "Foo Bar"],
                    "email": ["Foo bar", "Bar", "Foo Bar"],
                    "bank": [np.nan, "abc", "xyz"],
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
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
