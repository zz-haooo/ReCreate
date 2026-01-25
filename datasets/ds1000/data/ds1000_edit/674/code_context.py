import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        input = data
        tf.compat.v1.disable_eager_execution()
        ds = tf.data.Dataset.from_tensor_slices(input)
        ds = ds.flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices([x, x + 1, x + 2])
        )
        element = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
        result = []
        with tf.compat.v1.Session() as sess:
            for _ in range(9):
                result.append(sess.run(element))
        return result

    def define_test_input(test_case_id):
        if test_case_id == 1:
            tf.compat.v1.disable_eager_execution()
            input = [10, 20, 30]
        elif test_case_id == 2:
            tf.compat.v1.disable_eager_execution()
            input = [20, 40, 60]
        return input

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert result == ans
        return 1
    except:
        return 0


exec_context = r"""
import tensorflow as tf
input = test_input
tf.compat.v1.disable_eager_execution()
def f(input):
[insert]
result = f(input)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
