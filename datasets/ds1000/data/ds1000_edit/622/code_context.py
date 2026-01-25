import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    labels = ["Walking", "Talking", "Sleeping", "Working"]
    sizes = [23, 45, 12, 20]
    colors = ["red", "blue", "green", "yellow"]
    plt.pie(sizes, colors=colors, labels=labels, textprops={"weight": "bold"})
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
        assert len(ax.texts) == 4
        for t in ax.texts:
            assert "bold" in t.get_fontweight()
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
labels = ["Walking", "Talking", "Sleeping", "Working"]
sizes = [23, 45, 12, 20]
colors = ["red", "blue", "green", "yellow"]
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
