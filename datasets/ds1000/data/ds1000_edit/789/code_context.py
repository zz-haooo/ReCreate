import numpy as np
import copy
import scipy.integrate


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            N0 = 1
            time_span = [0, 10]
        return N0, time_span

    def generate_ans(data):
        _a = data
        N0, time_span = _a

        def dN1_dt(t, N1):
            input = 1 - np.cos(t) if 0 < t < 2 * np.pi else 0
            return -100 * N1 + input

        sol = scipy.integrate.solve_ivp(
            fun=dN1_dt,
            t_span=time_span,
            y0=[
                N0,
            ],
        )
        return sol.y

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import scipy.integrate
import numpy as np
N0, time_span = test_input
[insert]
result = sol.y
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
