import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, List = data
        df2 = df.iloc[List].reindex().reset_index(drop=True)
        return (df2.Type != df.Type).sum()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Col1": [1, 4, 7, 10, 13, 16],
                    "Col2": [2, 5, 8, 11, 14, 17],
                    "Col3": [3, 6, 9, 12, 15, 18],
                    "Type": [1, 1, 2, 2, 3, 3],
                }
            )
            List = np.random.permutation(len(df))
        return df, List

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert result == ans
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df, List = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
