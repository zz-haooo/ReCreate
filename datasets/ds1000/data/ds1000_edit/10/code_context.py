import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        if len(df.columns) == 1:
            if df.values.size == 1:
                return df.values[0][0]
            return df.values.squeeze()
        grouped = df.groupby(df.columns[0])
        d = {k: generate_ans(t.iloc[:, 1:]) for k, t in grouped}
        return d

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "name": ["A", "A", "B", "C", "B", "A"],
                    "v1": ["A1", "A2", "B1", "C1", "B2", "A2"],
                    "v2": ["A11", "A12", "B12", "C11", "B21", "A21"],
                    "v3": [1, 2, 3, 4, 5, 6],
                }
            )
        return df

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
