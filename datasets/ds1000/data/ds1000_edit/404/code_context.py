import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            index = ["x", "y"]
            columns = ["a", "b", "c"]
        return index, columns

    def generate_ans(data):
        _a = data
        index, columns = _a
        dtype = [("a", "int32"), ("b", "float32"), ("c", "float32")]
        values = np.zeros(2, dtype=dtype)
        df = pd.DataFrame(values, index=index)
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
index, columns = test_input
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
