import math

import numpy as np
import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertModel

from utils.utils import rematch


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = Linear(config.hidden_size, config.hidden_size)
        self.key = Linear(config.hidden_size, config.hidden_size)
        self.value = Linear(config.hidden_size, config.hidden_size)
        self.attention_activation = nn.ReLU()
        self.attention_epsilon = 1e10

    def forward(self, input_ids, mask):
        q = self.query(input_ids)
        k = self.key(input_ids)
        v = self.value(input_ids)

        q = self.attention_activation(q)
        k = self.attention_activation(k)
        v = self.attention_activation(v)

        e = torch.matmul(q, k.transpose(2, 1))
        e -= self.attention_epsilon * (1.0 - mask)
        a = torch.softmax(e, -1)
        v_o = torch.matmul(a, v)
        v_o += input_ids
        return v_o


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        # cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)
        outputs = outputs / std  # (b, s, h)
        outputs = outputs * weight + bias
        return outputs


class SpatialDropout(nn.Module):
    """
    对字级别的向量进行丢弃
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), 1, input.size(2))

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output


def get_entity(args, entity_heads_logits, entity_tails_logits, attention_masks, batch_offsets):
    entity_heads_logits = torch.sigmoid(entity_heads_logits)
    entity_tails_logits = torch.sigmoid(entity_tails_logits)

    entity_heads_logits = entity_heads_logits.detach().cpu().numpy()
    entity_tails_logits = entity_tails_logits.detach().cpu().numpy()
    attention_masks = attention_masks.cpu().numpy()

    batch_size = entity_heads_logits.shape[0]
    pred_triple_sets = []
    for index in range(batch_size):
        mapping = rematch(batch_offsets[index])
        text_attention_mask = attention_masks[index].reshape(-1, 1)
        entity_heads_logit = entity_heads_logits[index] * text_attention_mask
        entity_tails_logit = entity_tails_logits[index] * text_attention_mask

        entity_heads, entity_tails = np.where(
            entity_heads_logit > 0.5), np.where(entity_tails_logit > 0.5)
        subjects = []
        for head, head_type in zip(*entity_heads):
            for tail, tail_type in zip(*entity_tails):

                if head >= len(mapping) or tail >= len(mapping):
                    break
                if head <= tail and head_type == tail_type:
                    if head_type == 0:
                        subjects.append((head, tail))
                    else:
                        pass
                    break
        pred_triple_sets.append(subjects)
    return pred_triple_sets
    # 可能 预测的实体为空；


class RelEntityModel(nn.Module):
    def __init__(self, args, hidden_size) -> None:
        super().__init__()
        self.args = args
        pretrain_path = args.pretrain_path
        relation_size = args.relation_number
        self.bert = BertModel.from_pretrained(pretrain_path, cache_dir="./bertbaseuncased")
        self.entity_heads_out = nn.Linear(hidden_size, 2)  # 预测subjects,objects的头部位置
        self.entity_tails_out = nn.Linear(hidden_size, 2)  # 预测subjects,objects的尾部位置
        # self.rels_out = nn.Linear(hidden_size * 2, relation_size)  # 关系预测
        self.rels_out = nn.Linear(1536, relation_size)  # 关系预测

        if self.args.lastfour:
            self.keep_rels_out = nn.Linear(hidden_size*4, relation_size)  # 关系预测
        else:
            self.keep_rels_out = nn.Linear(hidden_size, relation_size)  # 关系预测
        self.birnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = 12
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.attention = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 1),
            nn.Softmax(dim=1)
        )

    def masked_avgpool(self, sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        pooler_output = torch.matmul(score.unsqueeze(1), sent).squeeze(1)
        return pooler_output

    def forward(self, input_ids, attention_masks, token_type_ids, batch_offsets):
        # 文本编码
        bert_output = self.bert(input_ids, attention_masks, token_type_ids, output_hidden_states=True)
        last_hidden_state = bert_output[0]
        all_hidden_size = bert_output[2]

        # last_hidden_state:cls 1x768,
        # hidden_state:token 31x768,
        # all_hidden_states":12x31x768

        # last_hidden_size = self.words_dropout(last_hidden_size)
        # print(last_hidden_state.size())
        # print(self.args.avg_pool)
        # print(self.args.lstm_pool)
        if self.args.avg_pool:
            pooler_output = self.masked_avgpool(last_hidden_state, attention_masks)
        elif self.args.lstm_pool:
            output, (hidden_last, cn) = self.birnn(last_hidden_state)
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            pooler_output = torch.cat([hidden_last_L, hidden_last_R], dim=-1)  # 64x1536
            # loguru.logger.info(pooler_output.size)
        elif self.args.second2last:
            all_hidden_states = torch.stack(bert_output[2])
            second_to_last_layer = 11
            pooler_output = all_hidden_states[second_to_last_layer, :, 0]
        elif self.args.lastfour:
            all_hidden_states = torch.stack(bert_output[2])
            concatenate_pooling = torch.cat(
                (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
            )
            pooler_output = concatenate_pooling[:, 0]
        elif self.args.att_pool:
            cls_outputs = torch.stack(
                [self.dropout(layer) for layer in all_hidden_size[-12:]], dim=0
            )
            # print(torch.softmax(self.layer_weights, dim=0).size())#[12]
            # print(torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1).size())# [12, 1, 1, 1]
            # print(cls_outputs.size())# [12, 64, 96, 768]
            cls_output = (
                    torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(
                0)  # 层间注意力

            pooler_output = torch.mean(
                torch.stack(
                    [torch.sum(self.attention(self.high_dropout(cls_output)) * cls_output, dim=1) for _ in range(5)],
                    # token注意力
                    dim=0,
                ),
                dim=0,
            )
        else:
            pooler_output = bert_output[1]

        # [batch,seq_len,2]
        pred_entity_heads = self.entity_heads_out(last_hidden_state)
        # [batch,seq_len,2]
        pred_entity_tails = self.entity_tails_out(last_hidden_state)
        # print("pred_entity_heads.requires_grad",pred_entity_heads.requires_grad)
        # print("pred_entity_tails.requires_grad",pred_entity_tails.requires_grad)

        pred_rels_raw = self.keep_rels_out(pooler_output)
        #
        pred_rels = []
        if self.args.use_split:  # 获取所有的hidden states进行加权平均
            subjects = get_entity(self.args, pred_entity_heads, pred_entity_tails, attention_masks, batch_offsets)
            # print(len(subjects))
            # print(subjects)
            # print("pred_entity_heads.requires_grad", pred_entity_heads.requires_grad)
            # print("pred_entity_tails.requires_grad", pred_entity_tails.requires_grad)
            for idx, sample_subjects in enumerate(subjects):  # 64 batch_size
                sample_rels = []
                if len(sample_subjects) > 0:
                    for entity in sample_subjects:
                        #
                        head, tail = entity
                        head_token_enmbedding = last_hidden_state[idx][head]
                        tail_token_enmbedding = last_hidden_state[idx][tail]
                        entity_emb = (head_token_enmbedding + tail_token_enmbedding) / 2

                        # print(head_token_enmbedding.size()) # 64, 81, 768
                        # print(tail_token_enmbedding.size()) # 64, 81, 768
                        # print(last_hidden_state.size()) # 64, 81, 768

                        # print(entity_emb.size()) # 768
                        # print(pooler_output[idx].size()) # 1536

                        entity_emb = torch.cat([entity_emb, pooler_output[idx]], dim=0)
                        # print(entity_emb.size()) # 2304

                        pred_rel = self.rels_out(entity_emb)  # 1x24
                        sample_rels.append(pred_rel)  # 24

                        # print(pred_rel.size())
                else:
                    pred_rel = self.keep_rels_out(pooler_output[idx])
                    sample_rels.append(pred_rel)
                    # print(pred_rel.size())
                # print(sample_rels[0].size())
                # print(torch.concat(sample_rels, dim=0).size())
                # print(torch.concat(sample_rels, dim=0).resize(len(sample_rels),24 ))
                # print(torch.concat(sample_rels, dim=1).size())
                # 1 3个实体 3x24
                # 2 2个实体 2x24
                # 3 1个实体 1x24
                pred_rels.append(torch.concat(sample_rels, dim=0).resize(len(sample_rels), 24))

        # print(pred_rels)
        # pred_rels = torch.concat(pred_rels, dim=0)
        # print(pred_rels.shape)

        if self.args.hidden_fuse:  # 获取所有的hidden states进行加权平均
            all_hidden_states = None
            j = 0
            for i, hiden_state in enumerate(all_hidden_size):
                print('self.args.hidden_fuse_layers', self.args.hidden_fuse_layers)
                print('i', i)
                if i in self.args.hidden_fuse_layers:
                    if all_hidden_states is None:
                        all_hidden_states = hiden_state * self.args.fuse_layers_weights[j]
                    else:
                        all_hidden_states += hiden_state * self.args.fuse_layers_weights[j]
                    j += 1
            # [batch_size,seq_len,hidden_size,layer_number]
            # all_hidden_states = torch.cat(all_hidden_states,dim=-1)
            # [batch_size,seq_len,hidden_size]
            # last_hidden_state = self.hidden_weight(all_hidden_states).squeeze(-1)
            last_hidden_state = all_hidden_states

        return pred_rels, pred_entity_heads, pred_entity_tails, last_hidden_state, pooler_output, pred_rels_raw


class ObjModel(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        relation_size = args.relation_number
        self.rels_out = nn.Linear(hidden_size, relation_size)  # 关系预测
        self.relu = nn.ReLU6()
        self.rel_feature = nn.Linear(hidden_size, hidden_size)
        # self.attention = BertSelfAttention(config)
        self.selfoutput = BertSelfOutput(config)
        self.attention = Attention(config)
        self.obj_head = nn.Linear(hidden_size, 1)
        self.words_dropout = SpatialDropout(0.1)
        self.conditionlayernormal = ConditionalLayerNorm(hidden_size, hidden_size * 2)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rel_sub_fuse = nn.Linear(hidden_size, hidden_size)
        self.relation_embedding = nn.Embedding(relation_size, hidden_size)

    def forward(self, relation, last_hidden_size, sub_head, sub_tail, attention_mask):
        """_summary_
        Args:
            relation (_type_): [batch_size,1] or [batch_size, rel_num]
            last_hidden_size (_type_): [batch_size,seq_len,hidden_size]
            sub_head (_type_): [batch_size,1] or [batch_size, rel_num]
            sub_tail (_type_): [batch_size,1] or [batch_size, rel_num]
        Returns:
            _type_: _description_
        """
        # last_hidden_size = self.words_dropout(last_hidden_size)
        last_hidden_size = self.dropout(last_hidden_size)
        # [batch_size,1,hidden_size]
        rel_feature = self.relation_embedding(relation)
        # [batch_size,1,hidden_size]
        rel_feature = self.relu(self.rel_feature(rel_feature))
        # [batch_size,1,1]
        sub_head = sub_head.unsqueeze(-1)
        # [batch_size,1,hidden_size]
        sub_head = sub_head.repeat(1, 1, self.hidden_size)
        # [batch_size,1,hidden_size]
        sub_head_feature = last_hidden_size.gather(1, sub_head)
        # [batch_size,1,1]
        sub_tail = sub_tail.unsqueeze(-1)
        # [batch_size,1,hidden_size]
        sub_tail = sub_tail.repeat(1, 1, self.hidden_size)
        # [batch_size,1,hidden_size]
        sub_tail_feature = last_hidden_size.gather(1, sub_tail)
        sub_feature = (sub_head_feature + sub_tail_feature) / 2
        if relation.shape[1] != 1:
            # [rel_num,1,hidden_size]
            rel_feature = rel_feature.transpose(1, 0)
            # [rel_num,1,hidden_size]
            sub_feature = sub_feature.transpose(1, 0)
        # feature = rel_feature+sub_feature
        # 将关系表征，subject表征进行线性变换
        feature = torch.cat([rel_feature, sub_feature], dim=-1)
        # feature = self.relu(self.rel_sub_fuse(feature))
        # feature = self.rel_sub_fuse(feature)

        # [batch_size,seq_len,hidden_size]
        hidden_size = self.conditionlayernormal(last_hidden_size, feature)
        # [batch_size,seq_len,hidden_size]
        obj_feature = hidden_size
        # obj_feature = last_hidden_size+rel_feature+sub_feature

        # bert self attention
        # attention_mask = self.expand_attention_masks(attention_mask)
        # hidden,*_ = self.attention(obj_feature,attention_mask)
        # hidden = self.selfoutput(hidden,feature)
        # 连接last_hidden_size残差的架构效果不好
        # hidden += last_hidden_size

        attention_mask = attention_mask.unsqueeze(1)
        hidden = self.attention(obj_feature, attention_mask)

        # [batch_size,seq_len,1]
        pred_obj_head = self.obj_head(hidden)
        # [batch_size,seq_len]
        pred_obj_head = pred_obj_head.squeeze(-1)
        return pred_obj_head, hidden

    def expand_attention_masks(self, attention_mask):
        batch_size, seq_length = attention_mask.shape
        causal_mask = attention_mask.unsqueeze(2).repeat(1, 1, seq_length) * attention_mask[:, None, :]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = causal_mask[:, None, :, :]
        extended_attention_mask = (1e-10) * (1 - extended_attention_mask)
        return extended_attention_mask


class TDEER(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrain_path = args.pretrain_path
        self.args = args
        config = BertConfig.from_pretrained(pretrain_path)
        hidden_size = config.hidden_size
        self.rel_entity_model = RelEntityModel(args, hidden_size)  # 关系 实体是被
        self.obj_model = ObjModel(args, config)

    def forward(self, input_ids, attention_masks, token_type_ids, relation=None, sub_head=None, sub_tail=None,
                batch_offsets=None):
        """_summary_
        Args:
            input_ids (_type_): [batch_size,seq_len]
            attention_masks (_type_): [batch_size,seq_len]
            token_type_ids (_type_): [batch_size,seq_len]
            relation (_type_, optional): [batch_size,1]. Defaults to None. subject 对应的关系(可以是正样本,也可也是负样本关系)
            sub_head (_type_, optional): [batch_size,1]. Defaults to None. subject 的head. 主要是为了预测object.如果是负样本关系,则预测不出object.
            sub_tail (_type_, optional): [batch_size,1]. Defaults to None. subject 的tail. 主要是为了预测object.如果是负样本关系,则预测不出object.
        Returns:
            _type_: _description_
        """
        pred_rels, pred_entity_heads, pred_entity_tails, last_hidden_state, pooler_output, pred_rels_raw = self.rel_entity_model(
            input_ids, attention_masks, token_type_ids, batch_offsets)
        pred_obj_head, obj_hidden = self.obj_model(relation, last_hidden_state, sub_head, sub_tail, attention_masks)
        return pred_rels, pred_entity_heads, pred_entity_tails, pred_obj_head, obj_hidden, pooler_output, pred_rels_raw
