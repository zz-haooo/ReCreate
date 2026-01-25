import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, list_of_my_columns = data
        df["Avg"] = df[list_of_my_columns].mean(axis=1)
        df["Min"] = df[list_of_my_columns].min(axis=1)
        df["Max"] = df[list_of_my_columns].max(axis=1)
        df["Median"] = df[list_of_my_columns].median(axis=1)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            data = {}
            for i in [chr(x) for x in range(65, 91)]:
                data["Col " + i] = np.random.randint(1, 100, 10)
            df = pd.DataFrame(data)
            list_of_my_columns = ["Col A", "Col E", "Col Z"]
        return df, list_of_my_columns

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
df, list_of_my_columns = test_input
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
