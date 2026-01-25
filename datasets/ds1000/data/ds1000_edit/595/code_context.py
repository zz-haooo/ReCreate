import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from matplotlib.ticker import PercentFormatter


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    data = [1000, 1000, 5000, 3000, 4000, 16000, 2000]
    plt.hist(data, weights=np.ones(len(data)) / len(data))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
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
        s = 0
        ax = plt.gca()
        plt.show()
        for rec in ax.get_children():
            if isinstance(rec, matplotlib.patches.Rectangle):
                s += rec._height
        assert s == 2.0
        for l in ax.get_yticklabels():
            assert "%" in l.get_text()
    return 1


exec_context = r"""
import numpy as np
import matplotlib.pyplot as plt
data = [1000, 1000, 5000, 3000, 4000, 16000, 2000]
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
