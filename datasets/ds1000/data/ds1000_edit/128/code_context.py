import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        l = int(0.2 * len(df))
        dfupdate = df.sample(l, random_state=0)
        dfupdate.ProductId = 0
        df.update(dfupdate)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "UserId": [1, 1, 1, 2, 3, 3],
                    "ProductId": [1, 4, 7, 4, 2, 1],
                    "Quantity": [6, 1, 3, 2, 7, 2],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "UserId": [1, 1, 1, 2, 3, 3, 1, 1, 4],
                    "ProductId": [1, 4, 7, 4, 2, 1, 5, 1, 4],
                    "Quantity": [6, 1, 3, 2, 7, 2, 9, 9, 6],
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
result = df
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
    assert "sample" in tokens
