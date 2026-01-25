import numpy as np
import copy
from scipy.optimize import minimize


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            I = np.array((20, 50, 50, 80))
        return I

    def generate_ans(data):
        _a = data

        def function(x):
            return -1 * (18 * x[0] + 16 * x[1] + 12 * x[2] + 11 * x[3])

        I = _a
        x0 = I
        cons = []
        steadystate = {"type": "eq", "fun": lambda x: x.sum() - I.sum()}
        cons.append(steadystate)

        def f(a):
            def g(x):
                return x[a]

            return g

        for t in range(4):
            cons.append({"type": "ineq", "fun": f(t)})
        out = minimize(function, x0, method="SLSQP", constraints=cons)
        x = out["x"]
        return x

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.optimize import minimize
def function(x):
    return -1*(18*x[0]+16*x[1]+12*x[2]+11*x[3])
I = test_input
x0=I
cons=[]
steadystate={'type':'eq', 'fun': lambda x: x.sum()-I.sum() }
cons.append(steadystate)
[insert]
out=minimize(function, x0, method="SLSQP", constraints=cons)
x=out["x"]
result = x
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
