import numpy as np
import copy
from scipy import ndimage


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            gen = np.random.RandomState(0)
            img = gen.poisson(2, size=(512, 512))
            img = ndimage.gaussian_filter(img.astype(np.double), (30, 30))
            img -= img.min()
            img /= img.max()
        return img

    def generate_ans(data):
        _a = data
        img = _a
        threshold = 0.75
        blobs = img > threshold
        labels, result = ndimage.label(blobs)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy import ndimage
img = test_input
threshold = 0.75
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
