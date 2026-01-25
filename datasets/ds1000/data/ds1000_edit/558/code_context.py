import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
    axes = axes.flatten()
    for ax in axes:
        ax.set_ylabel(r"$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$")
        ax.set_xlabel(r"$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$")
    plt.show()
    plt.clf()
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
    axes = axes.flatten()
    for ax in axes:
        ax.set_ylabel(r"$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$")
        ax.set_xlabel(r"$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$")
    plt.tight_layout()
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
        assert tuple(f.get_size_inches()) == (8, 6)
        assert f.subplotpars.hspace > 0.2
        assert f.subplotpars.wspace > 0.2
        assert len(f.axes) == 4
        for ax in f.axes:
            assert (
                ax.xaxis.get_label().get_text()
                == "$\\ln\\left(\\frac{x_a-x_d}{x_a-x_e}\\right)$"
            )
            assert (
                ax.yaxis.get_label().get_text()
                == "$\\ln\\left(\\frac{x_a-x_b}{x_a-x_c}\\right)$"
            )
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
axes = axes.flatten()
for ax in axes:
    ax.set_ylabel(r"$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$")
    ax.set_xlabel(r"$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$")
plt.show()
plt.clf()
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
