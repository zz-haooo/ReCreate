import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        x, y, z = data
        return tf.gather_nd(x, [y, z])

    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [[1, 2, 3], [4, 5, 6]]
            y = [0, 1]
            z = [1, 2]
            x = tf.constant(x)
            y = tf.constant(y)
            z = tf.constant(z)
        if test_case_id == 2:
            x = [[1, 2, 3], [4, 5, 6]]
            y = [0, 1]
            z = [1, 0]
            x = tf.constant(x)
            y = tf.constant(y)
            z = tf.constant(z)
        return x, y, z

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
x,y,z = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
