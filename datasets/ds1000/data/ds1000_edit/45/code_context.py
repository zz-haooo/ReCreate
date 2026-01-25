import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data

        def justify(a, invalid_val=0, axis=1, side="left"):
            if invalid_val is np.nan:
                mask = ~np.isnan(a)
            else:
                mask = a != invalid_val
            justified_mask = np.sort(mask, axis=axis)
            if (side == "up") | (side == "left"):
                justified_mask = np.flip(justified_mask, axis=axis)
            out = np.full(a.shape, invalid_val)
            if axis == 1:
                out[justified_mask] = a[mask]
            else:
                out.T[justified_mask.T] = a.T[mask.T]
            return out

        return pd.DataFrame(
            justify(df.values, invalid_val=np.nan, axis=1, side="right")
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                [[3, 1, 2], [1, 2, np.nan], [2, np.nan, np.nan]],
                columns=["0", "1", "2"],
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


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens and "while" not in tokens and "apply" not in tokens
