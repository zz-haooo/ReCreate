import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LogisticRegression


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            dataframe = pd.DataFrame(
                {
                    "Name": [
                        "T-Rex",
                        "Crocodile",
                        "Lion",
                        "Bear",
                        "Tiger",
                        "Hyena",
                        "Jaguar",
                        "Cheetah",
                        "KomodoDragon",
                    ],
                    "teethLength": [12, 4, 2.7, 3.6, 3, 0.27, 2, 1.5, 0.4],
                    "weight": [15432, 2400, 416, 600, 260, 160, 220, 154, 150],
                    "length": [40, 23, 9.8, 7, 12, 5, 5.5, 4.9, 8.5],
                    "hieght": [20, 1.6, 3.9, 3.35, 3, 2, 2.5, 2.9, 1],
                    "speed": [33, 8, 50, 40, 40, 37, 40, 70, 13],
                    "Calorie Intake": [
                        40000,
                        2500,
                        7236,
                        20000,
                        7236,
                        5000,
                        5000,
                        2200,
                        1994,
                    ],
                    "Bite Force": [12800, 3700, 650, 975, 1050, 1100, 1350, 475, 240],
                    "Prey Speed": [20, 30, 35, 0, 37, 20, 15, 56, 24],
                    "PreySize": [19841, 881, 1300, 0, 160, 40, 300, 185, 110],
                    "EyeSight": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "Smell": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "Class": [
                        "Primary Hunter",
                        "Primary Hunter",
                        "Primary Hunter",
                        "Primary Scavenger",
                        "Primary Hunter",
                        "Primary Scavenger",
                        "Primary Hunter",
                        "Primary Hunter",
                        "Primary Scavenger",
                    ],
                }
            )
            for column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(str).astype("category")
            dataframe = dataframe.drop(["Name"], axis=1)
            cleanup = {"Class": {"Primary Hunter": 0, "Primary Scavenger": 1}}
            dataframe.replace(cleanup, inplace=True)
        return dataframe

    def generate_ans(data):
        dataframe = data
        X = dataframe.iloc[:, 0:-1].astype(float)
        y = dataframe.iloc[:, -1]
        logReg = LogisticRegression()
        logReg.fit(X[:None], y)
        predict = logReg.predict(X)
        return predict

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
dataframe = test_input
[insert]
predict = logReg.predict(X)
result = predict
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
