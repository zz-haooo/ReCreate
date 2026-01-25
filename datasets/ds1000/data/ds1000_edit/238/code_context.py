import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df1, df2 = data
        df = pd.concat(
            [df1, df2.merge(df1[["id", "city", "district"]], how="left", on="id")],
            sort=False,
        ).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.strftime("%d-%b-%Y")
        return df.sort_values(by=["id", "date"]).reset_index(drop=True)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df1 = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "city": ["bj", "bj", "sh", "sh", "sh"],
                    "district": ["ft", "ft", "hp", "hp", "hp"],
                    "date": [
                        "2019/1/1",
                        "2019/1/1",
                        "2019/1/1",
                        "2019/1/1",
                        "2019/1/1",
                    ],
                    "value": [1, 5, 9, 13, 17],
                }
            )
            df2 = pd.DataFrame(
                {
                    "id": [3, 4, 5, 6, 7],
                    "date": [
                        "2019/2/1",
                        "2019/2/1",
                        "2019/2/1",
                        "2019/2/1",
                        "2019/2/1",
                    ],
                    "value": [1, 5, 9, 13, 17],
                }
            )
        if test_case_id == 2:
            df1 = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "city": ["bj", "bj", "sh", "sh", "sh"],
                    "district": ["ft", "ft", "hp", "hp", "hp"],
                    "date": [
                        "2019/2/1",
                        "2019/2/1",
                        "2019/2/1",
                        "2019/2/1",
                        "2019/2/1",
                    ],
                    "value": [1, 5, 9, 13, 17],
                }
            )
            df2 = pd.DataFrame(
                {
                    "id": [3, 4, 5, 6, 7],
                    "date": [
                        "2019/3/1",
                        "2019/3/1",
                        "2019/3/1",
                        "2019/3/1",
                        "2019/3/1",
                    ],
                    "value": [1, 5, 9, 13, 17],
                }
            )
        return df1, df2

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
df1, df2 = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
