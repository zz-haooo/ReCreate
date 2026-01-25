import numpy as np
import random
import copy
from scipy import stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:

            def poisson_simul(rate, T):
                time = random.expovariate(rate)
                times = [0]
                while times[-1] < T:
                    times.append(time + times[-1])
                    time = random.expovariate(rate)
                return times[1:]

            random.seed(42)
            rate = 1.0
            T = 100.0
            times = poisson_simul(rate, T)
        return rate, T, times

    def generate_ans(data):
        _a = data
        rate, T, times = _a
        result = stats.kstest(times, stats.uniform(loc=0, scale=T).cdf)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
from scipy import stats
import random
import numpy as np
def poisson_simul(rate, T):
    time = random.expovariate(rate)
    times = [0]
    while (times[-1] < T):
        times.append(time+times[-1])
        time = random.expovariate(rate)
    return times[1:]
rate, T, times = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
