import numpy as np
import copy
from sklearn.feature_extraction.text import CountVectorizer


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            corpus = [
                "We are looking for Java developer",
                "Frontend developer with knowledge in SQL and Jscript",
                "And this is the third one.",
                "Is this the first document?",
            ]
        return corpus

    def generate_ans(data):
        corpus = data
        vectorizer = CountVectorizer(
            stop_words="english",
            binary=True,
            lowercase=False,
            vocabulary=[
                "Jscript",
                ".Net",
                "TypeScript",
                "SQL",
                "NodeJS",
                "Angular",
                "Mongo",
                "CSS",
                "Python",
                "PHP",
                "Photoshop",
                "Oracle",
                "Linux",
                "C++",
                "Java",
                "TeamCity",
                "Frontend",
                "Backend",
                "Full stack",
                "UI Design",
                "Web",
                "Integration",
                "Database design",
                "UX",
            ],
        )
        X = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names_out()
        return feature_names, X

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_equal(result[0], ans[0])
        np.testing.assert_equal(result[1], ans[1])
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
corpus = test_input
[insert]
result = (feature_names, X)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
