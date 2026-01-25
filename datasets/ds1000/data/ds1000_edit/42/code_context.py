import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df.columns = np.concatenate([df.iloc[0, :2], df.columns[2:]])
        df = df.iloc[1:].reset_index(drop=True)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Nanonose": ["Sample type", "Water", "Water", "Water", "Water"],
                    "Unnamed: 1": ["Concentration", 9200, 9200, 9200, 4600],
                    "A": [
                        np.nan,
                        95.5,
                        94.5,
                        92.0,
                        53.0,
                    ],
                    "B": [np.nan, 21.0, 17.0, 16.0, 7.5],
                    "C": [np.nan, 6.0, 5.0, 3.0, 2.5],
                    "D": [np.nan, 11.942308, 5.484615, 11.057692, 3.538462],
                    "E": [np.nan, 64.134615, 63.205769, 62.586538, 35.163462],
                    "F": [np.nan, 21.498560, 19.658560, 19.813120, 6.876207],
                    "G": [np.nan, 5.567840, 4.968000, 5.192480, 1.641724],
                    "H": [np.nan, 1.174135, 1.883444, 0.564835, 0.144654],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Nanonose": ["type of Sample", "Water", "Water", "Water", "Water"],
                    "Unnamed: 1": ["concentration", 9200, 9200, 9200, 4600],
                    "A": [
                        np.nan,
                        95.5,
                        94.5,
                        92.0,
                        53.0,
                    ],
                    "B": [np.nan, 21.0, 17.0, 16.0, 7.5],
                    "C": [np.nan, 6.0, 5.0, 3.0, 2.5],
                    "D": [np.nan, 11.942308, 5.484615, 11.057692, 3.538462],
                    "E": [np.nan, 64.134615, 63.205769, 62.586538, 35.163462],
                    "F": [np.nan, 21.498560, 19.658560, 19.813120, 6.876207],
                    "G": [np.nan, 5.567840, 4.968000, 5.192480, 1.641724],
                    "H": [np.nan, 1.174135, 1.883444, 0.564835, 0.144654],
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
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
