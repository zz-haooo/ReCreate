import numpy as np
import copy
from scipy import misc
from scipy.ndimage import rotate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data_orig = misc.face()
            x0, y0 = 580, 300  # left eye; (xrot,yrot) should point there
            np.random.seed(42)
            angle = np.random.randint(1, 360)
        return data_orig, x0, y0, angle

    def generate_ans(data):
        _a = data
        data_orig, x0, y0, angle = _a

        def rot_ans(image, xy, angle):
            im_rot = rotate(image, angle)
            org_center = (np.array(image.shape[:2][::-1]) - 1) / 2.0
            rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.0
            org = xy - org_center
            a = np.deg2rad(angle)
            new = np.array(
                [
                    org[0] * np.cos(a) + org[1] * np.sin(a),
                    -org[0] * np.sin(a) + org[1] * np.cos(a),
                ]
            )
            return im_rot, new + rot_center

        data_rot, (xrot, yrot) = rot_ans(data_orig, np.array([x0, y0]), angle)
        return [data_rot, (xrot, yrot)]

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    res, (x1, y1) = result
    answer, (x_ans, y_ans) = ans
    assert np.allclose((x1, y1), (x_ans, y_ans))
    assert np.allclose(res, answer)
    return 1


exec_context = r"""
from scipy import misc
from scipy.ndimage import rotate
import numpy as np
data_orig, x0, y0, angle = test_input
[insert]
result = [data_rot, (xrot, yrot)]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
