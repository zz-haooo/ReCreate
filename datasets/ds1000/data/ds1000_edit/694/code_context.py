import tensorflow as tf
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        A, B = data
        return tf.constant(np.einsum("ikm, jkm-> ijk", A, B))

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            A = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
            B = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
        return A, B

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    def tensor_equal(a, b):
        if type(a) != type(b):
            return False
        if isinstance(a, type(tf.constant([]))) is not True:
            if isinstance(a, type(tf.Variable([]))) is not True:
                return False
        if a.shape != b.shape:
            return False
        if a.dtype != tf.float32:
            a = tf.cast(a, tf.float32)
        if b.dtype != tf.float32:
            b = tf.cast(b, tf.float32)
        if not tf.reduce_min(tf.cast(a == b, dtype=tf.int32)):
            return False
        return True

    try:
        assert tensor_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import tensorflow as tf
import numpy as np
A,B = test_input
[insert]
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
    assert "for" not in tokens and "while" not in tokens
