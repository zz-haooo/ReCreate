import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, prod_list = data
        for product in prod_list:
            df.loc[
                (df["product"] >= product[0]) & (df["product"] <= product[1]), "score"
            ] *= 10
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "product": [
                        1179160,
                        1066490,
                        1148126,
                        1069104,
                        1069105,
                        1160330,
                        1069098,
                        1077784,
                        1193369,
                        1179741,
                    ],
                    "score": [
                        0.424654,
                        0.424509,
                        0.422207,
                        0.420455,
                        0.414603,
                        0.168784,
                        0.168749,
                        0.168738,
                        0.168703,
                        0.168684,
                    ],
                }
            )
            products = [[1069104, 1069105], [1066489, 1066491]]
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "product": [
                        1179160,
                        1066490,
                        1148126,
                        1069104,
                        1069105,
                        1160330,
                        1069098,
                        1077784,
                        1193369,
                        1179741,
                    ],
                    "score": [
                        0.424654,
                        0.424509,
                        0.422207,
                        0.420455,
                        0.414603,
                        0.168784,
                        0.168749,
                        0.168738,
                        0.168703,
                        0.168684,
                    ],
                }
            )
            products = [
                [1069104, 1069105],
            ]
        return df, products

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
df, products = test_input
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
