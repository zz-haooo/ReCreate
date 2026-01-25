import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, row_list, column_list = data
        return df[column_list].iloc[row_list].mean(axis=0)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "a": [1, 1, 1, 1],
                    "b": [2, 2, 1, 0],
                    "c": [3, 3, 1, 0],
                    "d": [0, 4, 6, 0],
                    "q": [5, 5, 1, 0],
                }
            )
            row_list = [0, 2, 3]
            column_list = ["a", "b", "d"]
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "a": [1, 1, 1, 1],
                    "b": [2, 2, 1, 0],
                    "c": [3, 3, 1, 0],
                    "d": [0, 4, 6, 0],
                    "q": [5, 5, 1, 0],
                }
            )
            row_list = [0, 1, 3]
            column_list = ["a", "c", "q"]
        return df, row_list, column_list

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_series_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df, row_list, column_list = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens
