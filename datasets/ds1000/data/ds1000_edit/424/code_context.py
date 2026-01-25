import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            names = [
                "One",
                "Two",
                "Three",
                "Four",
                "Five",
                "Six",
                "Seven",
                "Eight",
                "Nine",
                "Ten",
                "Eleven",
                "Twelve",
                "Thirteen",
                "Fourteen",
                "Fifteen",
            ]
            times = [
                pd.Timestamp("2019-01-22 18:12:00"),
                pd.Timestamp("2019-01-22 18:13:00"),
                pd.Timestamp("2019-01-22 18:14:00"),
                pd.Timestamp("2019-01-22 18:15:00"),
                pd.Timestamp("2019-01-22 18:16:00"),
            ]
            df = pd.DataFrame(
                np.random.randint(10, size=(15 * 5, 4)),
                index=pd.MultiIndex.from_product(
                    [names, times], names=["major", "timestamp"]
                ),
                columns=list("colu"),
            )
        return names, times, df

    def generate_ans(data):
        _a = data
        names, times, df = _a
        result = df.values.reshape(15, 5, 4).transpose(0, 2, 1)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == np.ndarray
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
names, times, df = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
