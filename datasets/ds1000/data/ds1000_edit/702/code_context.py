import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        a = data
        return tf.argmax(a, axis=1)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = tf.constant(
                [
                    [0.3232, -0.2321, 0.2332, -0.1231, 0.2435, 0.6728],
                    [0.2323, -0.1231, -0.5321, -0.1452, 0.5435, 0.1722],
                    [0.9823, -0.1321, -0.6433, 0.1231, 0.023, 0.0711],
                ]
            )
        if test_case_id == 2:
            a = tf.constant(
                [
                    [0.3232, -0.2321, 0.2332, -0.1231, 0.2435],
                    [0.2323, -0.1231, -0.5321, -0.1452, 0.5435],
                    [0.9823, -0.1321, -0.6433, 0.1231, 0.023],
                ]
            )
        return a

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
a = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
