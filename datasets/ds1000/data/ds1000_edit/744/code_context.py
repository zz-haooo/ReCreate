import numpy as np
import copy
import tokenize, io
import scipy.ndimage


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            square = np.zeros((32, 32))
            square[10:-10, 10:-10] = np.random.randint(1, 255, size=(12, 12))
            np.random.seed(12)
            x, y = (32 * np.random.random((2, 20))).astype(int)
            square[x, y] = np.random.randint(1, 255, size=(20,))
        return square

    def generate_ans(data):
        _a = data
        square = _a

        def filter_isolated_cells(array, struct):
            filtered_array = np.copy(array)
            id_regions, num_ids = scipy.ndimage.label(filtered_array, structure=struct)
            id_sizes = np.array(
                scipy.ndimage.sum(array, id_regions, range(num_ids + 1))
            )
            area_mask = id_sizes == 1
            filtered_array[area_mask[id_regions]] = 0
            return filtered_array

        arr = np.sign(square)
        filtered_array = filter_isolated_cells(arr, struct=np.ones((3, 3)))
        square = np.where(filtered_array == 1, square, 0)
        return square

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.ndimage
square = test_input
[insert]
result = square
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
    assert "while" not in tokens and "for" not in tokens
