import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, columns = data
        return df.loc[df["c"] > 0.5, columns]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            df = pd.DataFrame(np.random.rand(4, 5), columns=list("abcde"))
            columns = ["b", "e"]
        return df, columns

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_array_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
def f(df, columns):
[insert]
df, columns = test_input
result = f(df, columns)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
