import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        to_delete = ["2020-02-17", "2020-02-18"]
        df = df[~(df.index.strftime("%Y-%m-%d").isin(to_delete))]
        df.index = df.index.strftime("%d-%b-%Y %A")
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Date": [
                        "2020-02-15 15:30:00",
                        "2020-02-16 15:31:00",
                        "2020-02-17 15:32:00",
                        "2020-02-18 15:33:00",
                        "2020-02-19 15:34:00",
                    ],
                    "Open": [2898.75, 2899.25, 2898.5, 2898.25, 2898.5],
                    "High": [2899.25, 2899.75, 2899, 2899.25, 2899.5],
                    "Low": [2896.5, 2897.75, 2896.5, 2897.75, 2898.25],
                    "Last": [2899.25, 2898.5, 2898, 2898, 2898.75],
                    "Volume": [1636, 630, 1806, 818, 818],
                    "# of Trades": [862, 328, 562, 273, 273],
                    "OHLC Avg": [2898.44, 2898.81, 2898, 2898.31, 2898.62],
                    "HLC Avg": [2898.33, 2898.67, 2897.75, 2898.33, 2898.75],
                    "HL Avg": [2897.88, 2898.75, 2897.75, 2898.5, 2898.75],
                    "Delta": [-146, 168, -162, -100, -100],
                    "HiLodiff": [11, 8, 10, 6, 6],
                    "OCdiff": [-2, 3, 2, 1, 1],
                    "div_Bar_Delta": [1, 2, -1, -1, -1],
                }
            )
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
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
