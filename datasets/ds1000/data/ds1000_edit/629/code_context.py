import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    df = sns.load_dataset("penguins")[["bill_length_mm", "species", "sex"]]
    sns.catplot(
        x="sex", col="species", y="bill_length_mm", data=df, kind="bar", sharey=False
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
        for ax in f.axes:
            assert ax.get_xlabel() == "sex"
            assert len(ax.patches) == 2
        assert f.axes[0].get_ylabel() == "bill_length_mm"
        assert len(f.axes[0].get_yticks()) != len(
            f.axes[1].get_yticks()
        ) or not np.allclose(f.axes[0].get_yticks(), f.axes[1].get_yticks())
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = sns.load_dataset("penguins")[["bill_length_mm", "species", "sex"]]
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
