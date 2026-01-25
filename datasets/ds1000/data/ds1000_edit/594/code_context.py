import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    df = pd.DataFrame(
        np.random.randn(50, 4),
        index=pd.date_range("1/1/2000", periods=50),
        columns=list("ABCD"),
    )
    df = df.cumsum()
    df.plot(style=".-")
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
        assert ax.get_lines()[0].get_linestyle() != "None"
        assert ax.get_lines()[0].get_marker() != "None"
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.DataFrame(
    np.random.randn(50, 4),
    index=pd.date_range("1/1/2000", periods=50),
    columns=list("ABCD"),
)
df = df.cumsum()
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
