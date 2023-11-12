import argparse
import json
import os

import loguru
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import TDEERDataset, collate_fn, collate_fn_val
from model import TDEER
from utils.loss_func import MLFocalLoss, BCEFocalLoss
from utils.utils import rematch
from utils.utils import update_arguments
from utils.adv_utils import FGM,EMA
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

log_writer = SummaryWriter('./log')


# ======================================
# config:args parse and load config
# ======================================

def parser_args():
    parser = argparse.ArgumentParser(description='各个模型公共参数')
    parser.add_argument('--model_type', default="tdeer",
                        type=str, help='定义模型类型', choices=['tdeer'])
    # parser.add_argument('--pretrain_path', type=str, default="luyaojie/uie-base-en", help='定义预训练模型路径')
    parser.add_argument('--pretrain_path', type=str, default="pretrained_models/bert-base-uncased",
                        help='定义预训练模型路径')
    parser.add_argument('--data_dir', type=str, default="data/NYT", help='定义数据集路径')
    parser.add_argument('--lr', default=2e-5, type=float, help='specify the learning rate')
    parser.add_argument('--bert_lr', default=2e-5, type=float, help='specify the learning rate for bert layer')
    parser.add_argument('--other_lr', default=2e-4, type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=20, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=64, type=int, help='specify the batch size')
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

    return args


args = parser_args()
print(args)

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

obj_loss = nn.BCEWithLogitsLoss(reduction="none")
focal_loss = MLFocalLoss()
b_focal_loss = BCEFocalLoss(alpha=0.25, gamma=2)
threshold = 0.5
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = args
loss_weight = args.loss_weight

rel_loss = nn.MultiLabelSoftMarginLoss()
entity_head_loss = nn.BCEWithLogitsLoss(reduction="none")
entity_tail_loss = nn.BCEWithLogitsLoss(reduction="none")
model = TDEER(args)
# print(model)
model.to(device)


def build_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']

    # param_optimizer = list(model.named_parameters())
    # print(list(model.named_parameters()))
    # layer_names = [name for name, param in model.named_parameters()]
    # print(layer_names)

    optimizer_grouped_parameters = [
        # bert
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                    and any(en in n for en, ep in model.rel_entity_model.bert.named_parameters())],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                    and any(en in n for en, ep in model.rel_entity_model.bert.named_parameters())],
         'weight_decay': 0.0, 'lr': args.bert_lr},

        # other
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                    and not any(en in n for en, ep in model.rel_entity_model.bert.named_parameters())],
         'weight_decay': args.weight_decay, 'lr': args.other_lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                    and not any(en in n for en, ep in model.rel_entity_model.bert.named_parameters())],
         'weight_decay': 0.0, 'lr': args.other_lr}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.eps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.train_steps * args.warmup_ratio,
                                                num_training_steps=args.train_steps)
    return optimizer, scheduler


def compute_kl_loss(p, q, pad_mask=None):
    """计算两个hidden states的 KL散度
    Args:
        p ([type]): [description]
        q ([type]): [description]
        pad_mask ([type], optional): [description]. Defaults to None.
    Returns:
        [type]: [description]
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask.type(torch.bool), 0.)
        q_loss.masked_fill_(pad_mask.type(torch.bool), 0.)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

rel_label_map = {
    'ORG': {
        "0": "/business/company/advisors",
        "1": "/business/company/founders",
        "2": "/business/company/industry",
        "3": "/business/company/major_shareholders",
        "4": "/business/company/place_founded",
        "16": "/people/person/ethnicity",  #
        "22": "/sports/sports_team/location",
    },
    'PER': {
        "5": "/business/company_shareholder/major_shareholder_of",
        "6": "/business/person/company",
        "12": "/people/deceased_person/place_of_death",
        "13": "/people/ethnicity/geographic_distribution",
        "14": "/people/ethnicity/people",
        "15": "/people/person/children",
        "17": "/people/person/nationality",
        "18": "/people/person/place_lived",
        "19": "/people/person/place_of_birth",
        "20": "/people/person/profession",
        "21": "/people/person/religion",
    },
    'LOC': {
        "7": "/location/administrative_division/country",
        "8": "/location/country/administrative_divisions",
        "9": "/location/country/capital",
        "10": "/location/location/contains",
        "11": "/location/neighborhood/neighborhood_of",
        "23": "/sports/sports_team_location/teams"
    }
}

def train_one(model, batch, rel_loss, entity_head_loss, entity_tail_loss, obj_loss):
    batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_entity_heads, batch_entity_tails, batch_rels, \
        batch_sample_subj_head, batch_sample_subj_tail, batch_sample_rel, batch_sample_obj_heads, batch_triple_sets, batch_text_masks = batch

    batch_tokens = batch_tokens.to(device)
    batch_attention_masks = batch_attention_masks.to(device)
    batch_segments = batch_segments.to(device)
    batch_entity_heads = batch_entity_heads.to(device)
    batch_entity_tails = batch_entity_tails.to(device)
    batch_rels = batch_rels.to(device)
    batch_sample_subj_head = batch_sample_subj_head.to(device)
    batch_sample_subj_tail = batch_sample_subj_tail.to(device)
    batch_sample_rel = batch_sample_rel.to(device)
    batch_sample_obj_heads = batch_sample_obj_heads.to(device)
    batch_text_masks = batch_text_masks.to(device)

    output = model(
        batch_tokens, batch_attention_masks, batch_segments,
        relation=batch_sample_rel, sub_head=batch_sample_subj_head,
        sub_tail=batch_sample_subj_tail,
        batch_offsets=batch_offsets
    )

    pred_rels, pred_entity_heads, pred_entity_tails, pred_obj_head, obj_hidden, last_hidden_size,relations_logits_raw  = output

    loss = 0
    rel_loss = rel_loss(relations_logits_raw, batch_rels)
    total_loss = 0
    # for idx,pred_rel in enumerate(pred_rels):
    #     # batch_rels = torch.mask(entity_types, 1)
    #     # print(pred_rel.size())
    #     # print(batch_rels.size())
    #     for entity_rel in pred_rel:
    #         total_loss += rel_loss(entity_rel, batch_rels[idx])
    # print(pred_rels)

    rel_loss += focal_loss(relations_logits_raw, batch_rels)
    loss += loss_weight[0] * total_loss

    batch_text_mask = batch_text_masks.reshape(-1, 1)

    pred_entity_heads = pred_entity_heads.reshape(-1, 2)
    batch_entity_heads = batch_entity_heads.reshape(-1, 2)
    entity_head_loss = entity_head_loss(pred_entity_heads, batch_entity_heads)
    entity_head_loss = (entity_head_loss * batch_text_mask).sum() / batch_text_mask.sum()
    loss += loss_weight[1] * entity_head_loss

    pred_entity_tails = pred_entity_tails.reshape(-1, 2)
    batch_entity_tails = batch_entity_tails.reshape(-1, 2)
    entity_tail_loss = entity_tail_loss(pred_entity_tails, batch_entity_tails)
    entity_tail_loss = (entity_tail_loss * batch_text_mask).sum() / batch_text_mask.sum()
    loss += loss_weight[2] * entity_tail_loss

    pred_obj_head = pred_obj_head.reshape(-1, 1)
    batch_sample_obj_heads = batch_sample_obj_heads.reshape(-1, 1)
    obj_loss = obj_loss(pred_obj_head, batch_sample_obj_heads)
    obj_loss += b_focal_loss(pred_obj_head, batch_sample_obj_heads)
    obj_loss = (obj_loss * batch_text_mask).sum() / batch_text_mask.sum()
    loss += loss_weight[3] * obj_loss
    return loss, obj_hidden, last_hidden_size


def train_epoch(model, epoch, optimizer, scheduler,fgm,ema):
    model.train()
    loguru.logger.info(f"training at {epoch}")
    losses = []
    tqdm_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"training epoch:\t {epoch}")
    for batch in tqdm_bar:
        loss, obj_hidden, last_hidden_size = train_one(
            model, batch,
            rel_loss, entity_head_loss, entity_tail_loss, obj_loss
        )
        if epoch>10:
            if args.is_rdrop:
                loss_2, obj_hidden_2, last_hidden_size_2 = train_one(
                    model, batch,
                    rel_loss, entity_head_loss, entity_tail_loss, obj_loss
                )
                loss = (loss + loss_2) / 2
                # obj_kl_loss = compute_kl_loss(obj_hidden,obj_hidden_2)
                hidden_size_kl_loss = compute_kl_loss(last_hidden_size, last_hidden_size_2)
                kl_loss = hidden_size_kl_loss
                loss = loss + 5 * kl_loss

            # print(loss)
            losses.append(loss.item())
            loss.backward()
        if epoch>10:
            ##对抗训练
            fgm.attack()
            loss_adv, _,_ = train_one(
                model, batch,
                rel_loss, entity_head_loss, entity_tail_loss, obj_loss
            )
            loss_adv.backward()
            fgm.restore()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        tqdm_bar.set_postfix_str(f'loss: {loss.item():.4f}')
        ema.update()

def validation_step(model, batch):
    batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_triple_sets, batch_triples_index_set, batch_text_masks = batch

    batch_tokens = batch_tokens.to(device)
    batch_attention_masks = batch_attention_masks.to(device)
    batch_segments = batch_segments.to(device)
    batch_text_masks = batch_text_masks.to(device)

    relations_logits, entity_heads_logits, entity_tails_logits, last_hidden_state, pooler_output,relations_logits_raw = model.rel_entity_model(
        batch_tokens, batch_attention_masks, batch_segments,batch_offsets)

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
    # print(len(batch_texts))
    # print(len(pred_triple_sets))
    # print(len(batch_triple_sets))
    return batch_texts, pred_triple_sets, batch_triple_sets


def validation_epoch_end(epoch, outputs):
    preds, targets = [], []
    texts = []
    # print(outputs[0])
    # for text, pred, target in zip(outputs):
    #     preds.extend(pred)
    #     targets.extend(target)
    #     texts.extend(text)
    texts, preds, targets = outputs
    correct = 0
    predict = 0
    total = 0
    orders = ['subject', 'relation', 'object']

    # log_dir = [log.log_dir for log in self.loggers if hasattr(log, "log_dir")][0]
    log_dir = '.'
    os.makedirs(os.path.join(log_dir, "output"), exist_ok=True)
    writer = open(os.path.join(log_dir, "output", 'val_output_{}.json'.format(epoch)), 'w')
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

    epoch += 1
    real_acc = round(correct / predict, 5) if predict != 0 else 0
    real_recall = round(correct / total, 5)
    real_f1 = round(2 * (real_recall * real_acc) / (real_recall + real_acc), 5) if (real_recall + real_acc) != 0 else 0
    # print("tot", float(total))
    # print("cor", float(correct))
    # print("pred", float(predict))
    # print("recall", float(real_recall))
    # print("acc", float(real_acc))
    # print("f1", float(real_f1))

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


def decode_entity(text: str, mapping, start: int, end: int):
    s = mapping[start]
    e = mapping[end]
    s = 0 if not s else s[0]
    e = len(text) - 1 if not e else e[-1]
    entity = text[s: e + 1]
    return entity


def valid_epoch(model, epoch,ema):
    ema.apply_shadow()
    model.eval()
    loguru.logger.info(f"validing at {epoch}")
    epoch_texts, epoch_pred_triple_sets, epoch_triple_sets = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            batch_texts, pred_triple_sets, batch_triple_sets = validation_step(model, batch)
            epoch_texts.extend(batch_texts)
            epoch_pred_triple_sets.extend(pred_triple_sets)
            epoch_triple_sets.extend(batch_triple_sets)
        # print(len(epoch_texts))
        # print(len(epoch_pred_triple_sets))
        # print(len(epoch_triple_sets))
        outputs = (epoch_texts, epoch_pred_triple_sets, epoch_triple_sets)
        validation_epoch_end(epoch, outputs)


print(args.is_train)
for epoch in range(num_epochs):
    optimizer, scheduler = build_optimizer(args, model)
    fgm = FGM(model)

    ema = EMA(model, 0.995)
    ema.register()

    if args.is_train:
        train_epoch(model, epoch, optimizer, scheduler,fgm,ema)
        torch.save(model.state_dict(), f"output/model_epoch{epoch}.bin")
    if args.is_valid:
        model.load_state_dict(torch.load(f"output/model_epoch{epoch}.bin"))
        valid_epoch(model, epoch,ema)
        ema.restore()