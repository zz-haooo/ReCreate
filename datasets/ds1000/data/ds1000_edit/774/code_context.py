import numpy as np
import copy
import scipy.fft as sf


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            N = 8
        elif test_case_id == 2:
            np.random.seed(42)
            N = np.random.randint(4, 15)
        return N

    def generate_ans(data):
        _a = data
        N = _a
        result = sf.dct(np.eye(N), axis=0, norm="ortho")
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.fft as sf
N = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
