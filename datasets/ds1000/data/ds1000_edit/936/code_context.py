import torch
import copy
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from torch import nn


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            input_Tensor = torch.LongTensor([1, 2, 3, 4, 5, 6, 7])
        return input_Tensor

    def generate_ans(data):
        input_Tensor = data
        model = Word2Vec(
            sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4
        )
        weights = torch.FloatTensor(model.wv.vectors)
        embedding = nn.Embedding.from_pretrained(weights)
        embedded_input = embedding(input_Tensor)
        return embedded_input

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        torch.testing.assert_close(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
input_Tensor = test_input
word2vec = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
[insert]
result = embedded_input
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
