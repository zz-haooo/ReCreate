import numpy as np
import pandas as pd
import copy
import tokenize, io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = pd.DataFrame(
                [
                    [
                        "Salut comment tu vas",
                        "Hey how are you today",
                        "I am okay and you ?",
                    ]
                ]
            ).T
            data.columns = ["test"]
        return data

    def generate_ans(data):
        data = data
        pipe = Pipeline([("tf_idf", TfidfVectorizer()), ("nmf", NMF())])
        pipe.fit_transform(data.test)
        tf_idf_out = pipe.named_steps["tf_idf"].transform(data.test)
        return tf_idf_out

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_allclose(result.toarray(), ans.toarray())
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
data = test_input
pipe = Pipeline([
    ("tf_idf", TfidfVectorizer()),
    ("nmf", NMF())
])
[insert]
result = tf_idf_out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "TfidfVectorizer" not in tokens
