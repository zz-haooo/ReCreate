import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    a, b = 1, 1
    c, d = 3, 4
    plt.axline((a, b), (c, d))
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.savefig("ans.png", bbox_inches="tight")
    plt.close()
    return None, None


def exec_test(result, ans):
    code_img = np.array(Image.open("output.png"))
    oracle_img = np.array(Image.open("ans.png"))
    sample_image_stat = code_img.shape == oracle_img.shape and np.allclose(
        code_img, oracle_img
    )
    if not sample_image_stat:
        ax = plt.gca()
        assert len(ax.get_lines()) == 1
        assert isinstance(ax.get_lines()[0], matplotlib.lines.AxLine)
        assert ax.get_xlim()[0] == 0 and ax.get_xlim()[1] == 5
        assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] == 5
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
a, b = 1, 1
c, d = 3, 4
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
