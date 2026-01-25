import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        time = df.time.tolist()
        car = df.car.tolist()
        nearest_neighbour = []
        euclidean_distance = []
        for i in range(len(df)):
            n = 0
            d = np.inf
            for j in range(len(df)):
                if (
                    df.loc[i, "time"] == df.loc[j, "time"]
                    and df.loc[i, "car"] != df.loc[j, "car"]
                ):
                    t = np.sqrt(
                        ((df.loc[i, "x"] - df.loc[j, "x"]) ** 2)
                        + ((df.loc[i, "y"] - df.loc[j, "y"]) ** 2)
                    )
                    if t < d:
                        d = t
                        n = df.loc[j, "car"]
            nearest_neighbour.append(n)
            euclidean_distance.append(d)
        return pd.DataFrame(
            {
                "time": time,
                "car": car,
                "nearest_neighbour": nearest_neighbour,
                "euclidean_distance": euclidean_distance,
            }
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            time = [0, 0, 0, 1, 1, 2, 2]
            x = [216, 218, 217, 280, 290, 130, 132]
            y = [13, 12, 12, 110, 109, 3, 56]
            car = [1, 2, 3, 1, 3, 4, 5]
            df = pd.DataFrame({"time": time, "x": x, "y": y, "car": car})
        if test_case_id == 2:
            time = [0, 0, 0, 1, 1, 2, 2]
            x = [219, 219, 216, 280, 290, 130, 132]
            y = [15, 11, 14, 110, 109, 3, 56]
            car = [1, 2, 3, 1, 3, 4, 5]
            df = pd.DataFrame({"time": time, "x": x, "y": y, "car": car})
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        result.euclidean_distance = np.round(result.euclidean_distance, 2)
        ans.euclidean_distance = np.round(ans.euclidean_distance, 2)
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
