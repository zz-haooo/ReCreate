import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return pd.pivot_table(
            df, values=["D", "E"], index=["B"], aggfunc={"D": np.max, "E": np.min}
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(1)
            df = pd.DataFrame(
                {
                    "A": ["one", "one", "two", "three"] * 6,
                    "B": ["A", "B", "C"] * 8,
                    "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                    "D": np.random.randn(24),
                    "E": np.random.randn(24),
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
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
