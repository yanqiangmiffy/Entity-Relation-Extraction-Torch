#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: quincy qiang
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2023/11/11 13:10
"""
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../pretrained_models/bert-base-uncased')
print(tokenizer)

token_ids = [
    101, 4404, 14327, 20201, 2307, 26469, 1025, 2036, 2012, 22759, 2267, 1010, 4698, 8943, 2571, 1011, 2006,
    1011, 6842, 1010, 1050, 1012, 1061, 1012, 1010, 2251, 1015, 1011, 15476, 1012, 102
]

text = "Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug ."
tokens = tokenizer.convert_ids_to_tokens(token_ids[1:-1])
print(text)
print(tokens)
print(" ".join(tokens))
tokens = tokenizer.convert_ids_to_tokens(token_ids[1:-1],skip_special_tokens=True)
print(tokens)

import numpy as np

sentences = []  # 存储结果的列表
line = 'Nanostructured Pt-alloy electrocatalysts for PEM fuel cell oxygen reduction reaction'

tokens = line.strip().split(' ')  # 按照空格进行分词

subwords = list(map(tokenizer.tokenize, tokens))
"""
[['Nan', '##ost', '##ru', '##cture', '##d'], ['P', '##t', '-', 'alloy'], ['electro', '##cat', '##aly', '##sts'], ['for'], ['P', '##EM'], ['fuel'], ['cell'], ['oxygen'], ['reduction'], ['reaction']]
"""
subword_lengths = list(map(len, subwords))
# [5, 4, 4, 1, 2, 1, 1, 1, 1, 1]

subwords = ['[CLS]'] + [item for indices in subwords for item in indices]
# ['[CLS]', 'Nan', '##ost', '##ru', '##cture', '##d', 'P', '##t', '-', 'alloy', 'electro', '##cat', '##aly', '##sts', ...]
# cls直接加到句首，并且列表

token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
# 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响

sentences.append((tokenizer.convert_tokens_to_ids(subwords), token_start_idxs))
# 存入结果，这里可以携程循环用于data loader中

print(sentences)