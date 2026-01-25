import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    column_labels = list("ABCD")
    row_labels = list("WXYZ")
    data = np.random.rand(4, 4)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    ax.xaxis.tick_top()
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
        assert ax.xaxis._major_tick_kw["tick2On"]
        assert ax.xaxis._major_tick_kw["label2On"]
        assert not ax.xaxis._major_tick_kw["tick1On"]
        assert not ax.xaxis._major_tick_kw["label1On"]
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
import numpy as np
column_labels = list("ABCD")
row_labels = list("WXYZ")
data = np.random.rand(4, 4)
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
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
