import pandas as pd
import io
import copy
from scipy import integrate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            string = """
    Time                      A
    2017-12-18-19:54:40   -50187.0
    2017-12-18-19:54:45   -60890.5
    2017-12-18-19:54:50   -28258.5
    2017-12-18-19:54:55    -8151.0
    2017-12-18-19:55:00    -9108.5
    2017-12-18-19:55:05   -12047.0
    2017-12-18-19:55:10   -19418.0
    2017-12-18-19:55:15   -50686.0
    2017-12-18-19:55:20   -57159.0
    2017-12-18-19:55:25   -42847.0
    """
            df = pd.read_csv(io.StringIO(string), sep="\s+")
        return df

    def generate_ans(data):
        _a = data
        df = _a
        df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d-%H:%M:%S")
        df = df.set_index("Time")
        integral_df = df.rolling("25S").apply(integrate.trapz)
        return integral_df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans, check_dtype=False)
    return 1


exec_context = r"""
import pandas as pd
import io
from scipy import integrate
df = test_input
[insert]
result = integral_df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
