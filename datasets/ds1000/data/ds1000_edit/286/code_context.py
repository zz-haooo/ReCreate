import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        cols = list(df)
        Mode = df.mode(axis=1)
        df["frequent"] = df["bit1"].astype(object)
        for i in df.index:
            df.at[i, "frequent"] = []
        for i in df.index:
            for col in list(Mode):
                if pd.isna(Mode.loc[i, col]) == False:
                    df.at[i, "frequent"].append(Mode.loc[i, col])
            df.at[i, "frequent"] = sorted(df.at[i, "frequent"])
            df.loc[i, "freq_count"] = (
                df[cols].iloc[i] == df.loc[i, "frequent"][0]
            ).sum()
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "bit1": [0, 2, 4],
                    "bit2": [0, 2, 0],
                    "bit3": [3, 0, 4],
                    "bit4": [3, 0, 4],
                    "bit5": [0, 2, 4],
                    "bit6": [3, 0, 5],
                }
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
for i in df.index:
    df.at[i, 'frequent'] = sorted(df.at[i, 'frequent'])
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
