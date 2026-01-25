import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
        y = df["Date"].dt.year
        m = df["Date"].dt.month
        w = df["Date"].dt.weekday
        df["Count_d"] = df.groupby("Date")["Date"].transform("size")
        df["Count_m"] = df.groupby([y, m])["Date"].transform("size")
        df["Count_y"] = df.groupby(y)["Date"].transform("size")
        df["Count_w"] = df.groupby(w)["Date"].transform("size")
        df["Count_Val"] = df.groupby(["Date", "Val"])["Val"].transform("size")
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            d = {
                "Date": [
                    "1/1/18",
                    "1/1/18",
                    "1/1/18",
                    "2/1/18",
                    "3/1/18",
                    "1/2/18",
                    "1/3/18",
                    "2/1/19",
                    "3/1/19",
                ],
                "Val": ["A", "A", "B", "C", "D", "A", "B", "C", "D"],
            }
            df = pd.DataFrame(data=d)
        if test_case_id == 2:
            d = {
                "Date": [
                    "1/1/19",
                    "1/1/19",
                    "1/1/19",
                    "2/1/19",
                    "3/1/19",
                    "1/2/19",
                    "1/3/19",
                    "2/1/20",
                    "3/1/20",
                ],
                "Val": ["A", "A", "B", "C", "D", "A", "B", "C", "D"],
            }
            df = pd.DataFrame(data=d)
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
