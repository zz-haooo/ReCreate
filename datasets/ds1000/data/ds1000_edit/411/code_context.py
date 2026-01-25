import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.ones((3, 3))
            low_index = -1
            high_index = 2
        elif test_case_id == 2:
            a = np.ones((5, 5)) * 2
            low_index = 1
            high_index = 6
        elif test_case_id == 3:
            a = np.ones((5, 5))
            low_index = 2
            high_index = 7
        return a, low_index, high_index

    def generate_ans(data):
        _a = data
        a, low_index, high_index = _a

        def fill_crop(img, pos, crop):
            img_shape, pos, crop_shape = (
                np.array(img.shape),
                np.array(pos),
                np.array(crop.shape),
            )
            end = pos + crop_shape
            crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
            crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
            crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
            pos = np.clip(pos, a_min=0, a_max=img_shape)
            end = np.clip(end, a_min=0, a_max=img_shape)
            img_slices = (slice(low, high) for low, high in zip(pos, end))
            crop[tuple(crop_slices)] = img[tuple(img_slices)]
            return crop

        result = fill_crop(
            a,
            [low_index, low_index],
            np.zeros((high_index - low_index, high_index - low_index)),
        )
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, low_index, high_index = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
