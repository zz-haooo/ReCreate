import numpy as np
import pandas as pd
import copy
from scipy import stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            np.random.seed(17)
            df = pd.DataFrame(
                {
                    "NUM1": np.random.randn(50) * 100,
                    "NUM2": np.random.uniform(0, 1, 50),
                    "NUM3": np.random.randint(100, size=50),
                    "CAT1": ["".join(np.random.choice(LETTERS, 1)) for _ in range(50)],
                    "CAT2": [
                        "".join(
                            np.random.choice(
                                ["pandas", "r", "julia", "sas", "stata", "spss"], 1
                            )
                        )
                        for _ in range(50)
                    ],
                    "CAT3": [
                        "".join(
                            np.random.choice(
                                [
                                    "postgres",
                                    "mysql",
                                    "sqlite",
                                    "oracle",
                                    "sql server",
                                    "db2",
                                ],
                                1,
                            )
                        )
                        for _ in range(50)
                    ],
                }
            )
        return df

    def generate_ans(data):
        _a = data
        df = _a
        df = df[
            (np.abs(stats.zscore(df.select_dtypes(exclude="object"))) < 3).all(axis=1)
        ]
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans, check_dtype=False, atol=1e-5)
    return 1


exec_context = r"""
from scipy import stats
import pandas as pd
import numpy as np
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
df = test_input
[insert]
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
