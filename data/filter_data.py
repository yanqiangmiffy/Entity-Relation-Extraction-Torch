#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: quincy qiang
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/1/7 15:59
"""
import pandas as pd


def split_dataset(dataset_name='NYT'):
    df = pd.read_json(f'{dataset_name}/dev_triples.json')
    print(df.columns)

    df['triple_list_lens'] = df['triple_list'].apply(lambda x: len(x))

    print(df)
    print(df['triple_list_lens'].describe())
    df['triple_list_lens'] = df['triple_list_lens'].apply(lambda x: 5 if x >= 5 else x)
    print(df['triple_list_lens'].value_counts())

    for idx, group in df.groupby(by="triple_list_lens"):
        print(idx)
        print(group)
        group.to_json(f"{dataset_name}/dev_triples_N{idx}.json", orient="records",indent=4)

split_dataset(dataset_name='NYT')
split_dataset(dataset_name='WebNLG')