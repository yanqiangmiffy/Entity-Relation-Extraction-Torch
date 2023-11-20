#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: quincy qiang
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2023/11/16 12:18
"""
import pandas as pd

webnlg=pd.read_json('data/WebNLG/train_triples.json')
print(webnlg)