import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df.set_index("Time", inplace=True)
        df_group = df.groupby(pd.Grouper(level="Time", freq="3T"))["Value"].agg("sum")
        df_group.dropna(inplace=True)
        df_group = df_group.to_frame().reset_index()
        return df_group

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Time": [
                        "2015-04-24 06:38:49",
                        "2015-04-24 06:39:19",
                        "2015-04-24 06:43:49",
                        "2015-04-24 06:44:18",
                        "2015-04-24 06:44:48",
                        "2015-04-24 06:45:18",
                        "2015-04-24 06:47:48",
                        "2015-04-24 06:48:18",
                        "2015-04-24 06:50:48",
                        "2015-04-24 06:51:18",
                        "2015-04-24 06:51:48",
                        "2015-04-24 06:52:18",
                        "2015-04-24 06:52:48",
                        "2015-04-24 06:53:48",
                        "2015-04-24 06:55:18",
                        "2015-04-24 07:00:47",
                        "2015-04-24 07:01:17",
                        "2015-04-24 07:01:47",
                    ],
                    "Value": [
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                        0.023844,
                        0.019075,
                    ],
                }
            )
            df["Time"] = pd.to_datetime(df["Time"])
        if test_case_id == 2:
            np.random.seed(4)
            df = pd.DataFrame(
                {
                    "Time": [
                        "2015-04-24 06:38:49",
                        "2015-04-24 06:39:19",
                        "2015-04-24 06:43:49",
                        "2015-04-24 06:44:18",
                        "2015-04-24 06:44:48",
                        "2015-04-24 06:45:18",
                        "2015-04-24 06:47:48",
                        "2015-04-24 06:48:18",
                        "2015-04-24 06:50:48",
                        "2015-04-24 06:51:18",
                        "2015-04-24 06:51:48",
                        "2015-04-24 06:52:18",
                        "2015-04-24 06:52:48",
                        "2015-04-24 06:53:48",
                        "2015-04-24 06:55:18",
                        "2015-04-24 07:00:47",
                        "2015-04-24 07:01:17",
                        "2015-04-24 07:01:47",
                    ],
                    "Value": np.random.random(18),
                }
            )
            df["Time"] = pd.to_datetime(df["Time"])
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
