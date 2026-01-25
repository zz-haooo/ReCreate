import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return df.sort_index(level="time")

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "VIM": [
                        -0.158406,
                        0.039158,
                        -0.052608,
                        0.157153,
                        0.206030,
                        0.132580,
                        -0.144209,
                        -0.093910,
                        -0.166819,
                        0.097548,
                        0.026664,
                        -0.008032,
                    ]
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("TGFb", 0.1, 2),
                        ("TGFb", 1, 2),
                        ("TGFb", 10, 2),
                        ("TGFb", 0.1, 24),
                        ("TGFb", 1, 24),
                        ("TGFb", 10, 24),
                        ("TGFb", 0.1, 48),
                        ("TGFb", 1, 48),
                        ("TGFb", 10, 48),
                        ("TGFb", 0.1, 6),
                        ("TGFb", 1, 6),
                        ("TGFb", 10, 6),
                    ],
                    names=["treatment", "dose", "time"],
                ),
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
