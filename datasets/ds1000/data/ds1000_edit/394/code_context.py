import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            string = "[[ 0.5544  0.4456], [ 0.8811  0.1189]]"
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(5, 6)
            string = str(a).replace("\n", ",")
        return string

    def generate_ans(data):
        _a = data
        string = _a
        a = np.array(np.matrix(string.replace(",", ";")))
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
string = test_input
[insert]
result = a
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
