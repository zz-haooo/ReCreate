import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        a, b = data
        if len(a) < len(b):
            for i in range(len(a), len(b)):
                a.loc[i] = [np.nan for _ in range(len(list(a)))]
        elif len(a) > len(b):
            for i in range(len(b), len(a)):
                b.loc[i] = [np.nan for _ in range(len(list(a)))]
        return pd.DataFrame(
            np.rec.fromarrays((a.values, b.values)).tolist(),
            columns=a.columns,
            index=a.index,
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["one", "two"])
            b = pd.DataFrame(
                np.array([[5, 6], [7, 8], [9, 10]]), columns=["one", "two"]
            )
        if test_case_id == 2:
            a = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["one", "two"])
            b = pd.DataFrame(np.array([[7, 8], [9, 10]]), columns=["one", "two"])
        return a, b

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
a,b = test_input
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
    assert "for" not in tokens and "while" not in tokens
