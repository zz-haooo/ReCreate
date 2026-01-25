import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return df.columns[df.iloc[0, :].fillna("Nan") == df.iloc[8, :].fillna("Nan")]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            df = pd.DataFrame(
                np.random.randint(0, 20, (10, 10)).astype(float),
                columns=["c%d" % d for d in range(10)],
            )
            df.where(
                np.random.randint(0, 2, df.shape).astype(bool), np.nan, inplace=True
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_index_equal(ans, result)
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
