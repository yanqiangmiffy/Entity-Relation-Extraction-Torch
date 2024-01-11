#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: quincy qiang
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/1/7 15:19
"""
import argparse
import json
import os
import random

import loguru
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TDEERDataset, collate_fn, collate_fn_val
from utils.utils import rematch
from utils.utils import update_arguments

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

log_writer = SummaryWriter('./log')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(42)

# ======================================
# config:args parse and load config
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parser_args(exp_name="tdeer_exp4_nyt_second2last", num_triples=1):
    parser = argparse.ArgumentParser(description='各个模型公共参数')
    parser.add_argument('--model_type', default=exp_name,
                        type=str, help='定义模型类型', choices=['tdeer'])
    parser.add_argument('--pretrain_path', type=str, default="pretrained_models/bert-base-uncased",
                        help='定义预训练模型路径')
    parser.add_argument('--lr', default=2e-5, type=float, help='specify the learning rate')
    parser.add_argument('--bert_lr', default=2e-5, type=float, help='specify the learning rate for bert layer')
    parser.add_argument('--other_lr', default=2e-4, type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=30, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=4, type=int, help='specify the batch size')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')
    parser.add_argument('--float16', default=False, type=bool, help='是否采用浮点16进行半精度计算')
    parser.add_argument('--grad_accumulations_steps', default=3, type=int, help='梯度累计步骤')

    # 不同学习率scheduler的参数
    parser.add_argument('--decay_rate', default=0.999, type=float, help='StepLR scheduler 相关参数')
    parser.add_argument('--decay_steps', default=100, type=int, help='StepLR scheduler 相关参数')
    parser.add_argument('--T_mult', default=1.0, type=float, help='CosineAnnealingWarmRestarts scheduler 相关参数')
    parser.add_argument('--rewarm_epoch_num', default=2, type=int,
                        help='CosineAnnealingWarmRestarts scheduler 相关参数')

    # 阶段
    parser.add_argument('--is_train', default=True, type=bool, help='是否训练')
    parser.add_argument('--is_valid', default=True, type=bool, help='是否评估')
    args = parser.parse_args()

    # 根据超参数文件更新参数
    config_file = os.path.join("config", "{}.yaml".format(args.model_type))
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    args = update_arguments(args, config['model_params'])
    args.config_file = config_file

    args.num_triples = num_triples
    # ======================================
    # dataset:load dataset
    # ======================================

    train_dataset = TDEERDataset(args, is_training=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
    val_dataset = TDEERDataset(args, is_training=False)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_val, batch_size=args.batch_size, shuffle=False)

    relation_number = train_dataset.relation_size
    args.relation_number = relation_number
    args.train_steps = len(train_dataset)
    args.warmup_ratio = 0.1
    args.weight_decay = 0.01
    args.eps = 1e-6
    args.threshold = 0.5
    loguru.logger.info(args.relation_number)
    loguru.logger.info(args.train_steps)

    # ======================================
    # model:initialize model
    # ======================================

    with open(os.path.join(args.data_dir, "rel2id.json"), 'r') as f:
        relation = json.load(f)
    id2rel = relation[0]

    return args, id2rel, train_dataloader, val_dataloader


def validation_step(model, batch, args, id2rel):
    batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_triple_sets, batch_triples_index_set, batch_text_masks = batch

    batch_tokens = batch_tokens.to(device)
    batch_attention_masks = batch_attention_masks.to(device)
    batch_segments = batch_segments.to(device)
    batch_text_masks = batch_text_masks.to(device)

    relations_logits_new, entity_heads_logits, entity_tails_logits, last_hidden_state, pooler_output, relations_logits_raw = model.rel_entity_model(
        batch_tokens, batch_attention_masks, batch_segments, batch_offsets)

    entity_heads_logits = torch.sigmoid(entity_heads_logits)
    entity_tails_logits = torch.sigmoid(entity_tails_logits)

    relations_logits = torch.sigmoid(relations_logits_raw)
    batch_size = entity_heads_logits.shape[0]
    entity_heads_logits = entity_heads_logits.cpu().numpy()
    entity_tails_logits = entity_tails_logits.cpu().numpy()
    relations_logits = relations_logits.cpu().numpy()
    batch_text_masks = batch_text_masks.cpu().numpy()

    pred_triple_sets = []
    for index in range(batch_size):
        mapping = rematch(batch_offsets[index])
        text = batch_texts[index]
        text_attention_mask = batch_text_masks[index].reshape(-1, 1)
        entity_heads_logit = entity_heads_logits[index] * text_attention_mask
        entity_tails_logit = entity_tails_logits[index] * text_attention_mask

        entity_heads, entity_tails = np.where(
            entity_heads_logit > args.threshold), np.where(entity_tails_logit > args.threshold)
        subjects = []
        entity_map = {}
        for head, head_type in zip(*entity_heads):
            for tail, tail_type in zip(*entity_tails):
                if head <= tail and head_type == tail_type:
                    if head >= len(mapping) or tail >= len(mapping):
                        break
                    entity = decode_entity(text, mapping, head, tail)
                    if head_type == 0:
                        subjects.append((entity, head, tail))
                    else:
                        entity_map[head] = entity
                    break

        triple_set = set()
        if len(subjects):
            # translating decoding
            relations = np.where(relations_logits[index] > args.threshold)[0].tolist()
            if relations:
                batch_sub_heads = []
                batch_sub_tails = []
                batch_rels = []
                batch_sub_entities = []
                batch_rel_types = []
                for (sub, sub_head, sub_tail) in subjects:
                    for rel in relations:
                        batch_sub_heads.append([sub_head])
                        batch_sub_tails.append([sub_tail])
                        batch_rels.append([rel])
                        batch_sub_entities.append(sub)
                        batch_rel_types.append(id2rel[str(rel)])
                batch_sub_heads = torch.tensor(
                    batch_sub_heads, dtype=torch.long, device=last_hidden_state.device)
                batch_sub_tails = torch.tensor(
                    batch_sub_tails, dtype=torch.long, device=last_hidden_state.device)
                batch_rels = torch.tensor(
                    batch_rels, dtype=torch.long, device=last_hidden_state.device)
                hidden = last_hidden_state[index].unsqueeze(0)
                attention_mask = batch_attention_masks[index].unsqueeze(0)
                batch_sub_heads = batch_sub_heads.transpose(1, 0)
                batch_sub_tails = batch_sub_tails.transpose(1, 0)
                batch_rels = batch_rels.transpose(1, 0)
                obj_head_logits, _ = model.obj_model(
                    batch_rels, hidden, batch_sub_heads, batch_sub_tails, attention_mask)
                obj_head_logits = torch.sigmoid(obj_head_logits)
                obj_head_logits = obj_head_logits.cpu().numpy()
                text_attention_mask = text_attention_mask.reshape(1, -1)
                for sub, rel, obj_head_logit in zip(batch_sub_entities, batch_rel_types, obj_head_logits):
                    obj_head_logit = obj_head_logit * text_attention_mask
                    for h in np.where(obj_head_logit > args.threshold)[1].tolist():
                        if h in entity_map:
                            obj = entity_map[h]
                            triple_set.add((sub, rel, obj))
        pred_triple_sets.append(triple_set)
    return batch_texts, pred_triple_sets, batch_triple_sets


def validation_epoch_end(outputs):
    texts, preds, targets = outputs
    correct = 0
    predict = 0
    total = 0
    orders = ['subject', 'relation', 'object']

    # log_dir = [log.log_dir for log in self.loggers if hasattr(log, "log_dir")][0]
    log_dir = '.'
    os.makedirs(os.path.join(log_dir, "output"), exist_ok=True)
    writer = open(os.path.join(log_dir, "output", 'val_output.json'.format()), 'w', encoding='utf-8')
    for text, pred, target in zip(*(texts, preds, targets)):
        pred = set([tuple(l) for l in pred])
        target = set([tuple(l) for l in target])
        correct += len(set(pred) & (target))
        predict += len(set(pred))
        total += len(set(target))
        new = [dict(zip(orders, triple)) for triple in pred - target]
        lack = [dict(zip(orders, triple)) for triple in target - pred]
        if len(new) or len(lack):
            result = json.dumps({
                'text': text,
                'golds': [
                    dict(zip(orders, triple)) for triple in target
                ],
                'preds': [
                    dict(zip(orders, triple)) for triple in pred
                ],
                'new': new,
                'lack': lack
            }, ensure_ascii=False)
            writer.write(result + '\n')
    writer.close()

    only_sub_rel_cor = 0
    only_sub_rel_pred = 0
    only_sub_rel_tot = 0
    for pred, target in zip(*(preds, targets)):
        pred = [list(l) for l in pred]
        pred = [(l[0], l[1]) for l in pred if len(l)]
        target = [(l[0], l[1]) for l in target]
        only_sub_rel_cor += len(set(pred).intersection(set(target)))
        only_sub_rel_pred += len(set(pred))
        only_sub_rel_tot += len(set(target))

    real_acc = round(only_sub_rel_cor / only_sub_rel_pred, 5) if only_sub_rel_pred != 0 else 0
    real_recall = round(only_sub_rel_cor / only_sub_rel_tot, 5)
    real_f1 = round(2 * (real_recall * real_acc) / (real_recall + real_acc), 5) if (real_recall + real_acc) != 0 else 0
    print("tot", only_sub_rel_tot)
    print("cor", only_sub_rel_cor)
    print("pred", only_sub_rel_pred)
    print("rec", real_recall)
    print("acc", real_acc)
    print("f1", real_f1)
    return {'rec': real_recall, 'pred': real_acc, 'f1': real_f1}


def decode_entity(text: str, mapping, start: int, end: int):
    s = mapping[start]
    e = mapping[end]
    s = 0 if not s else s[0]
    e = len(text) - 1 if not e else e[-1]
    entity = text[s: e + 1]
    return entity


def valid_epoch(exp_name="tdeer_exp4_nyt_second2last", numbers=5):
    print(exp_name, numbers)
    args, id2rel, train_dataloader, val_dataloader = parser_args(exp_name, numbers)
    if 'exp3' in exp_name:
        from model3 import TDEER
        model = TDEER(args)

    else:
        from model4 import TDEER
        model = TDEER(args)
    model.to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu"),strict=False)
    model.eval()
    epoch_texts, epoch_pred_triple_sets, epoch_triple_sets = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            batch_texts, pred_triple_sets, batch_triple_sets = validation_step(model, batch, args, id2rel)
            epoch_texts.extend(batch_texts)
            epoch_pred_triple_sets.extend(pred_triple_sets)
            epoch_triple_sets.extend(batch_triple_sets)
        outputs = (epoch_texts, epoch_pred_triple_sets, epoch_triple_sets)
        results = validation_epoch_end(outputs)
    results['exp_name'] = exp_name
    results['N'] = numbers
    print(results)
    return results


data = []
for exp in os.listdir('config'):
    exp = exp.split('.yaml')[0]
    for i in range(1, 6):
        try:
            results = valid_epoch(exp_name=exp, numbers=i)
            data.append(results)
        except:
            print("====报错====")
            print(exp, i)

pd.DataFrame(data).to_csv('results.csv', index=False)
