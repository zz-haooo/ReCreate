import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        FLAG = data
        return 1

    def define_test_input(test_case_id):
        if test_case_id == 1:
            FLAG = 114514
        return FLAG

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert result == ans
        return 1
    except:
        return 0


exec_context = r"""
import shutil
import os
if os.path.exists('my_model'):
    shutil.rmtree('my_model')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
FLAG = test_input
network_layout = []
for i in range(3):
    network_layout.append(8)
model = Sequential()
inputdim = 4
activation = 'relu'
outputdim = 2
opt='rmsprop'
epochs = 50
model.add(Dense(network_layout[0],
                name="Input",
                input_dim=inputdim,
                kernel_initializer='he_normal',
                activation=activation))
for numneurons in network_layout[1:]:
    model.add(Dense(numneurons,
                    kernel_initializer = 'he_normal',
                    activation=activation))
model.add(Dense(outputdim,
                name="Output",
                kernel_initializer="he_normal",
                activation="relu"))
model.compile(optimizer=opt,loss='mse',metrics=['mse','mae','mape'])
[insert]
try:
    assert os.path.exists("export")
    p = os.path.join("export", "1")
    assert os.path.exists(p)
    assert os.path.exists(os.path.join(p, "assets"))
    assert os.path.exists(os.path.join(p, "saved_model.pb"))
    p = os.path.join(p, "variables")
    assert os.path.exists(p)
    assert os.path.exists(os.path.join(p, "variables.data-00000-of-00001"))
    assert os.path.exists(os.path.join(p, "variables.index"))
    result = 1
except:
    result = 0
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "saved_model" in tokens
