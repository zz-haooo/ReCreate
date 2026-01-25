import numpy as np
import pandas as pd
import copy
from sklearn.cluster import KMeans


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "date": [
                        "2018-02-11",
                        "2018-02-12",
                        "2018-02-13",
                        "2018-02-14",
                        "2018-02-16",
                        "2018-02-21",
                        "2018-02-22",
                        "2018-02-24",
                        "2018-02-26",
                        "2018-02-27",
                        "2018-02-28",
                        "2018-03-01",
                        "2018-03-05",
                        "2018-03-06",
                    ],
                    "mse": [
                        14.34,
                        7.24,
                        4.5,
                        3.5,
                        12.67,
                        45.66,
                        15.33,
                        98.44,
                        23.55,
                        45.12,
                        78.44,
                        34.11,
                        23.33,
                        7.45,
                    ],
                }
            )
        return df

    def generate_ans(data):
        df = data
        kmeans = KMeans(n_clusters=2, n_init=10)
        labels = kmeans.fit_predict(df[["mse"]])
        return labels

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    ret = 0
    try:
        np.testing.assert_equal(result, ans)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_equal(result, 1 - ans)
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
df = test_input
[insert]
result = labels
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
