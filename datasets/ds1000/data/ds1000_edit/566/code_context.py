import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import seaborn as sns
from PIL import Image
import numpy as np


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    tips = sns.load_dataset("tips")
    sns.jointplot(
        x="total_bill", y="tip", data=tips, kind="reg", line_kws={"color": "green"}
    )
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
        f = plt.gcf()
        assert len(f.axes) == 3
        assert len(f.axes[0].get_lines()) == 1
        assert f.axes[0].get_xlabel() == "total_bill"
        assert f.axes[0].get_ylabel() == "tip"
        assert f.axes[0].get_lines()[0]._color in ["green", "g", "#008000"]
        for p in f.axes[1].patches:
            assert p.get_facecolor()[0] != 0
            assert p.get_facecolor()[2] != 0
        for p in f.axes[2].patches:
            assert p.get_facecolor()[0] != 0
            assert p.get_facecolor()[2] != 0
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import seaborn as sns
tips = sns.load_dataset("tips")
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
