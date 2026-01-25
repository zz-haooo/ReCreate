import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    x = np.arange(10)
    y = np.arange(10)
    plt.plot(x, y, label="1")
    plt.title("test title", fontsize=20)
    plt.xlabel("xlabel", fontsize=18)
    plt.ylabel("ylabel", fontsize=16)
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
        ylabel_font = ax.yaxis.get_label().get_fontsize()
        xlabel_font = ax.xaxis.get_label().get_fontsize()
        title_font = ax.title.get_fontsize()
        assert ylabel_font != xlabel_font
        assert title_font != xlabel_font
        assert title_font != ylabel_font
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.arange(10)
y = np.arange(10)
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
