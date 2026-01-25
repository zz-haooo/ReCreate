import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    a = [2.56422, 3.77284, 3.52623]
    b = [0.15, 0.3, 0.45]
    c = [58, 651, 393]
    fig, ax = plt.subplots()
    plt.scatter(a, b)
    for i, txt in enumerate(c):
        ax.annotate(txt, (a[i], b[i]))
    plt.savefig("ans.png", bbox_inches="tight")
    plt.close()
    return None, None


def exec_test(result, ans):
    c = [58, 651, 393]
    code_img = np.array(Image.open("output.png"))
    oracle_img = np.array(Image.open("ans.png"))
    sample_image_stat = code_img.shape == oracle_img.shape and np.allclose(
        code_img, oracle_img
    )
    if not sample_image_stat:
        ax = plt.gca()
        assert len(ax.texts) == 3
        for t in ax.texts:
            assert int(t.get_text()) in c
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a = [2.56422, 3.77284, 3.52623]
b = [0.15, 0.3, 0.45]
c = [58, 651, 393]
[insert]
plt.savefig('output.png', bbox_inches ='tight')
result = None
"""


def test_execution(solution: str):
    solution = "\n".join(filter(skip_plt_cmds, solution.split("\n")))
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
