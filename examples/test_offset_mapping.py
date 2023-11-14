#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: quincy qiang
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2023/11/11 14:17

https://blog.csdn.net/znsoft/article/details/124290977
https://blog.csdn.net/znevegiveup1/article/details/122737497
"""
from transformers import BertTokenizerFast

MODELNAME = "../pretrained_models/bert-base-uncased"

tokenizer = BertTokenizerFast.from_pretrained(MODELNAME)
text = "80 % of Americans believe seeking multiple opinions can help them make better choices,and for good reason."
text="Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug ."
tokens = tokenizer.tokenize(text, add_special_tokens=True)
outputs = tokenizer.encode_plus(text, return_offsets_mapping=True,
                                return_attention_mask=True,
                                add_special_tokens=True)  # add_special_tokens=True 添加 [cls] [sep]等标志
token_span = outputs["offset_mapping"]
print(len(tokens),tokens)
print(len(token_span),token_span)
# print(outputs.keys())

for offset_mapping in token_span:
    start,end=offset_mapping
    print(text[start:end],offset_mapping)

# print(outputs)