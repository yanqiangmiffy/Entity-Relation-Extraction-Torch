import argparse
import gc
import os
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from mixout import MixLinear
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from utils import FGM, RDropLoss, EMA

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
print(f"torch.__version__: {torch.__version__}")

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--model_name', type=str, default="F:/pretrained_models/roformer_chinese_base",
                    help='预训练模型名字')
parser.add_argument('--max_len', type=int, default=256, help='文本最大长度')
parser.add_argument('--trn_fold', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='分折个数')
parser.add_argument('--log_prefix', type=str, default="train", help='日志文件名称')
args = parser.parse_args()

print(args.trn_fold)
print(args.log_prefix)
print(args.max_len)


# ====================================================
# CFG:参数配置
# ====================================================
class Config:
    # 配置
    apex = False
    seed = 42  # 随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0  # 进程个数
    print_freq = 100  # 打印频率
    debug = False
    train = True
    predict = True

    # 预训练模型
    model_type = 'bert'
    model_name = args.model_name
    # model_type = 'roformer'
    # model_type = 'nezha'
    # model_name="junnyu/structbert-large-zh"
    # model_name = "hfl/chinese-pert-large-mrc"
    # model_name = "hfl/chinese-electra-180g-large-discriminator"
    # model_name = "junnyu/roformer_chinese_base"
    # model_name = "user_data/new_self_pretrained_model"
    # model_name = "F:/pretrained_models/nezha-large-zh"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 数据相关
    task_name = ''  # 比赛任务名称
    dataset_name = ''  # 数据集名称
    text_col = 'text'  # 文本列名
    target_col = 'label'  # label列名
    target_size = 2  # 类别个数
    max_len = 100  # 最大长度
    batch_size = 32  # 模型运行批处理大小
    n_fold = 5  # cv折数
    trn_folds = [0, 1, 2, 3, 4]  # 需要用的折数

    # 模型训练超参数
    epochs = 7  # 训练轮次
    lr = 2e-5  # 训练过程中的最大学习率
    eps = 1e-6
    betas = (0.9, 0.999)
    # warmup_proportion = 0.1  # 学习率预热比例
    num_warmup_steps = 0.1
    T_0 = 2  # CosineAnnealingWarmRestarts
    min_lr = 1e-6
    warmup_ratio = 0.1
    weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    gradient_accumulation_steps = 1  # 梯度累加
    max_grad_norm = 1.0  # 梯度剪切

    # 目录设置
    # 目录设置
    log_prefix = args.log_prefix  # 模型输出目录
    output_dir = './models2'  # 模型输出目录
    save_prefix = model_name.split("/")[-1]  # 保存文件前缀
    log_prefix = log_prefix + '_' + save_prefix

    # trick
    use_fgm = True  # fgm pgd
    use_pgd = False  # fgm pgd

    pgd_k = 3
    use_ema = True
    use_rdrop = True
    use_multidrop = False

    use_noisy = False
    use_mixout = False

    # 损失函数
    criterion = 'CrossEntropyLoss'  # - ['LabelSmoothing', 'FocalLoss','FocalLossPro','FocalCosineLoss', 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss']
    smoothing = 0.01
    scheduler = 'linear'  # ['linear', 'cosine']

    # score_type
    score_type = 'f1_macro'


CFG = Config()

os.makedirs(CFG.output_dir, exist_ok=True)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.seed)


def get_logger(filename=CFG.output_dir + f'/{CFG.log_prefix}'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()
train = pd.read_pickle('data/df_train.pkl')

print(train.isnull().sum())
print(train.shape)


def clean_txt(txt):
    try:
        txt = txt.replace(' ', '')
        txt = txt.replace('　', '')
        txt = txt.replace('\r', '')
        txt = txt.replace('\n', '')
        txt = txt.replace('“', '"')
        txt = txt.replace('”', '"')
        txt = txt.replace(',', '，')

        if txt[-1] == '.':
            txt[-1] == '。'
    except Exception as e:
        txt = txt.encode('utf-8', 'replace').decode('utf-8')
    txt = ''.join(txt.split())[:512]
    return txt


train['text'] = train['text'].apply(clean_txt)
print(train.shape)
print(train[CFG.target_col].nunique())
print(train[CFG.target_col].value_counts())
print(train.shape)

# token_lens = []
# for txt in tqdm(train.text):
#     tokens = CFG.tokenizer.encode(txt)
#     token_lens.append(len(tokens))
# print(pd.Series(token_lens).describe())
# print(pd.Series(token_lens).quantile([0.5, 0.8, 0.9, 0.99]))

# ===================
# CV SPLIT
# ===================

Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_col])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

# %%
if CFG.debug:
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        #         print(encoding['input_ids'])
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size, is_shuffle=False):
    ds = CustomDataset(
        texts=df[CFG.text_col].values,
        labels=df[CFG.target_col].values,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_shuffle,
        pin_memory=True,
        # num_workers=4  # windows多线程
    )


# class MeanPooling(nn.Module):
#     def __init__(self):
#         super(MeanPooling, self).__init__()
#
#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings
#
#
# class Custom_bert(nn.Module):
#     def __init__(self, model_name, n_classes):
#         super().__init__()
#
#         # config = BertConfig.from_pretrained(args.model_path)
#         self.config = AutoConfig.from_pretrained(model_name)
#         # self.bert = BertModel.from_pretrained(args.model_path, config=config)
#         self.bert = AutoModel.from_pretrained(model_name, config=self.config)
#
#         self.num_labels = n_classes
#         self.loss_fct = nn.CrossEntropyLoss()
#
#         self.pool = MeanPooling()
#
#         self.dropout_fc = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
#         self._init_weights(self.classifier)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         bert_out = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         last_hidden_states = bert_out[0]
#         feature = self.pool(last_hidden_states, attention_mask)
#
#         logits = self.classifier(self.dropout_fc(feature))
#         return logits
class Custom_bert(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        self.linear = nn.Linear(self.config.hidden_size, n_classes)
        self._init_weights(self.linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        logits = self.linear(mean_embeddings)
        return logits


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        epoch,
        fgm,
        ema,
        rdrop
):
    model = model.train()
    losses = []
    predictions = []
    real_values = []
    # correct_predictions = 0
    tqdm_bar = tqdm(data_loader, total=len(data_loader), desc=f"training epoch:\t {epoch}")
    for batch in tqdm_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        targets = batch["labels"].to(device)

        if CFG.use_rdrop:
            logits1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits2 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = rdrop(logits1, logits2, targets)
            # print(loss)
            # torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引（代表类别）
            _, preds = torch.max(logits1, dim=1)
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs, targets)
            # print(loss)
            # torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引（代表类别）
            _, preds = torch.max(outputs, dim=1)

        # correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()

        tqdm_bar.set_postfix_str(f'running training loss: {loss.item():.4f}')
        if CFG.use_fgm:
            ##对抗训练
            fgm.attack()
            adv_preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss_adv = loss_fn(adv_preds, targets)
            loss_adv.backward()
            fgm.restore()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if CFG.use_ema:
            ema.update()

        predictions.extend(preds)
        real_values.extend(targets)
    predictions = torch.stack(predictions).cpu().numpy()
    real_values = torch.stack(real_values).cpu().numpy()
    f1 = f1_score(predictions, real_values, average='binary')
    # f1 = accuracy_score(predictions, real_values)
    # LOGGER.info(f"epoch {epoch} training f1:{f1}")
    return f1, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, epoch, ema):
    if CFG.use_ema:
        ema.apply_shadow()
    model = model.eval()  # 验证预测模式
    losses = []
    prediction_probs = []
    predictions = []
    real_values = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc=f"evaluating:\t {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            _, preds = torch.max(outputs, dim=1)  # argmax
            probs = outputs.softmax(1)
            prediction_probs.extend(probs)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            predictions.extend(preds)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu().numpy()
    real_values = torch.stack(real_values).cpu().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().numpy()
    f1 = f1_score(predictions, real_values, average='binary')
    return f1, np.mean(losses), prediction_probs


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    # param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        # bert
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                    and any(en in n for en, ep in model.bert.named_parameters())],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                    and any(en in n for en, ep in model.bert.named_parameters())],
         'weight_decay': 0.0, 'lr': args.lr},

        # other
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                    and not any(en in n for en, ep in model.bert.named_parameters())],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                    and not any(en in n for en, ep in model.bert.named_parameters())],
         'weight_decay': 0.0, 'lr': args.lr}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.eps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps * args.warmup_ratio,
                                                num_training_steps=train_steps)
    return optimizer, scheduler


def train_cv():
    oof_df = pd.DataFrame()
    best_scores = []
    for fold in CFG.trn_folds:
        LOGGER.info(f'Fold {fold + 1}/{CFG.n_fold}')
        LOGGER.info('***' * 10)
        train_folds = train[train['fold'] != fold].reset_index(drop=True)
        valid_folds = train[train['fold'] == fold].reset_index(drop=True)
        train_data_loader = create_data_loader(train_folds, CFG.tokenizer, CFG.max_len, CFG.batch_size, True)
        val_data_loader = create_data_loader(valid_folds, CFG.tokenizer, CFG.max_len, CFG.batch_size, False)
        model = Custom_bert(CFG.model_name, CFG.target_size)
        model = model.to(CFG.device)

        if CFG.use_noisy:
            print("noisy tuning")
            # noisy tuning
            noise_lambda = 0.1
            for name, para in model.named_parameters():
                # if 'model' in name and 'bias' not in name and 'LayerNorm.weight' not in name:
                if 'model' in name:
                    model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)
        if CFG.use_mixout:
            print("use_mixout")
            # mixout mixes the parameters of the nn.Linear right after the nn.Dropout.
            for sup_module in model.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = MixLinear(
                            module.in_features, module.out_features, bias, target_state_dict["weight"], 0.1
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)

        # no_decay = ['bias', 'LayerNorm.weight']

        # param_optimizer = list(model.named_parameters())
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': CFG.weight_decay},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.0}
        # ]
        #
        # optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.lr, weight_decay=CFG.weight_decay, correct_bias=False)
        #
        # total_steps = len(train_data_loader) * CFG.epochs
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=int(CFG.num_warmup_steps * total_steps),
        #     num_training_steps=total_steps
        # )
        total_steps = CFG.epochs * len(train_data_loader)
        optimizer, scheduler = build_optimizer(CFG, model, total_steps)
        loss_fn = nn.CrossEntropyLoss().to(CFG.device)
        if CFG.use_rdrop:
            rdrop_loss = RDropLoss(loss_func=loss_fn)
        else:
            rdrop_loss = None

        if CFG.use_fgm:
            fgm = FGM(model)
        else:
            fgm = None

        if CFG.use_ema:
            ema = EMA(model, 0.995)
            ema.register()
        else:
            ema = None

        history = defaultdict(list)  # 记录10轮loss和acc
        best_accuracy = 0

        for epoch in range(CFG.epochs):

            # print(f'Epoch {epoch + 1}/{CFG.epochs}')
            # print('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                CFG.device,
                scheduler,
                epoch,
                fgm,
                ema,
                rdrop_loss
            )

            # LOGGER.info(f'epoch {epoch} Train loss {round(train_loss, 5)} f1 {round(train_acc, 5)}')

            val_acc, val_loss, val_probs = eval_model(
                model,
                val_data_loader,
                loss_fn,
                CFG.device,
                epoch,
                ema
            )

            LOGGER.info(
                f'epoch {epoch} [Train] loss {round(train_loss, 5)} f1 {round(train_acc, 5)} \t [Val]   loss {round(val_loss, 5)} f1 {round(val_acc, 5)} ')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                # torch.save(model.state_dict(), f'{CFG.output_dir}/{CFG.save_prefix}_best_model_fold{fold}.bin')
                torch.save({'model': model.state_dict(), 'predictions': val_probs},
                           f'{CFG.output_dir}/{CFG.save_prefix}_best_model_fold{fold}.bin')

                best_accuracy = val_acc
            if CFG.use_ema:
                # evaluate
                ema.restore()
        best_scores.append(best_accuracy)
        predictions = \
            torch.load(f'{CFG.output_dir}/{CFG.save_prefix}_best_model_fold{fold}.bin',
                       map_location=torch.device('cpu'))[
                'predictions']
        valid_folds[[str(c) for c in range(CFG.target_size)]] = predictions

        torch.cuda.empty_cache()
        gc.collect()

        oof_df = pd.concat([oof_df, valid_folds])

    print(best_scores)
    print(np.mean(best_scores))

    msg = "cv best_scores are: " + " ".join([str(score) for score in best_scores])
    LOGGER.info(msg)
    LOGGER.info(np.mean(best_scores))

    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_csv(f'{CFG.output_dir}/oof_df.csv', index=None)


def get_predictions(model, data_loader):
    model = model.eval()

    prediction_probs = []

    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader), desc="inference"):
            input_ids = d["input_ids"].to(CFG.device)
            attention_mask = d["attention_mask"].to(CFG.device)
            token_type_ids = d["token_type_ids"].to(CFG.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            probs = outputs.softmax(1)
            prediction_probs.extend(probs)
    prediction_probs = torch.stack(prediction_probs).cpu().numpy()
    return prediction_probs


def inference():
    test[CFG.target_col] = -1
    test_data_loader = create_data_loader(test, CFG.tokenizer, CFG.max_len, CFG.batch_size, False)
    probs = np.zeros((len(test), CFG.target_size))
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_folds:
            model = Custom_bert(model_name=CFG.model_name, n_classes=CFG.target_size)
            path = f'{CFG.output_dir}/{CFG.save_prefix}_best_model_fold{fold}.bin'
            state = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            model.eval()
            model.to(CFG.device)
            prediction_probs = get_predictions(model, test_data_loader)
            probs += prediction_probs / len(CFG.trn_folds)
    np.save(f'{CFG.output_dir}/{CFG.save_prefix}_probs.npy', probs)
    print(probs)
    labels = np.argmax(probs, axis=1)
    test[CFG.target_col] = labels
    test[['id', CFG.target_col, 'sentence_pair']].to_json(f'{CFG.output_dir}/{CFG.save_prefix}_submission.json',
                                                          orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    train_cv()
    inference()
