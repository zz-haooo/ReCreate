import numpy as np
import copy
from sklearn.feature_extraction.text import CountVectorizer


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            words = "Hello @friend, this is a good day. #good."
        elif test_case_id == 2:
            words = (
                "ha @ji me te no ru bu ru wa, @na n te ko to wa na ka tsu ta wa. wa ta shi da ke no mo na ri za, "
                "mo u to kku ni #de a t te ta ka ra"
            )
        return words

    def generate_ans(data):
        words = data
        count = CountVectorizer(
            lowercase=False, token_pattern="[a-zA-Z0-9$&+:;=@#|<>^*()%-]+"
        )
        vocabulary = count.fit_transform([words])
        feature_names = count.get_feature_names_out()
        return feature_names

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_equal(sorted(result), sorted(ans))
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
words = test_input
[insert]
result = feature_names
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
