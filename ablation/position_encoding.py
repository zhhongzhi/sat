import os
import torch
import random
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_vm',
                    type=str,
                    choices=["yes", "no"],
                    default='no')

parser.add_argument('--use_summary_row',
                    type=str,
                    choices=["yes", "no"],
                    default='no')


parser.add_argument('--only_low_vm',
                    type=str,
                    choices=["yes", "no"],
                    default='no')



parser.add_argument('--only_high_vm',
                    type=str,
                    choices=["yes", "no"],
                    default='no')




args = parser.parse_args()
print('use_vm', args.use_vm)
print('use_summary_row', args.use_summary_row)
print('only_low_vm', args.only_low_vm)
print('only_high_vm', args.only_high_vm)


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

from collections import defaultdict
# from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import random
import numpy as np

import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
import torch.nn as nn

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]




import torch.optim as optim
import sklearn

# from pytorch_pretrained_bert import BertModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer

# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

LL = 256
# LL = 512

# roberta_dp = 'roberta_large/'
# roberta_dp = 'roberta/'
roberta_dp = 'bert_base_multiligual_cased/'
# roberta_dp = 'bert_base_uncased/'
# roberta_dp = 'bert_large_uncased/'
pad_token = 1 if 'roberta' in roberta_dp else 0

if 'roberta' not in roberta_dp:
    RobertaModel = BertModel
    RobertaTokenizer = BertTokenizer



class DataManager:
    def __init__(self):
#         dp = '/home/hadoop-aipnlp/cephfs/data/aikg-tag/freeze_bert/'
        # self.tokenizer = BertTokenizer.from_pretrained(dp + 'bert-base-uncased')
        #         dp = ''
#         self.tokenizer = BertTokenizer.from_pretrained(dp + 'bert-base-uncased-vocab.txt')
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_dp)

        import pickle 
        if 'roberta' in roberta_dp or 'large' in roberta_dp or 'multi' in roberta_dp:
            tab_f = f"tables_and_count_info_{roberta_dp[:-1]}.cache"
            if os.path.exists(tab_f):
                print('load tables from cache', tab_f)
                self.tables = pickle.load(open(tab_f, 'rb'))
            else:
                self.tables = self.load_the_tables()  # 加载所有的表格
                pickle.dump(self.tables, open(tab_f, 'wb'))
        else:
            print('load tables from cache', 'tables_and_count_info.cache')
            self.tables = pickle.load(open('tables_and_count_info.cache', 'rb'))
            
        self.all_dat = self.load_all_the_dat()
        train_smps_ = self.get_dat('train')
        test_smps_ = self.get_dat('test')
        valid_smps_ = self.get_dat('val')
        train_smps = self.trans2ids(train_smps_)
        test_smps = self.trans2ids(test_smps_)
        valid_smps = self.trans2ids(valid_smps_)
#         train_smps, test_smps, valid_smps = pickle.load(open("samples.cache", 'rb'))

        self.train_dataset = MyDataSet(train_smps)
        self.dev_dataset = MyDataSet(valid_smps)
        self.test_dataset = MyDataSet(test_smps)
        
        self.simple_smps = MyDataSet(self.trans2ids(self.get_dat('simple')))
        self.complex_smps = MyDataSet(self.trans2ids(self.get_dat('complex')))


    def load_the_tables(self):
        dp = 'data/all_csv/'
        table_contents = {}
        for fnm in tqdm(os.listdir(dp)):
            table_contents[fnm] = self.load_a_table(dp + fnm)
        return table_contents

    def load_a_table(self, fnm):
        contents = []
        for ln in open(fnm).readlines():
            info = ln.strip("\n").split("#")
            info = [self.tokenizer.tokenize(i) for i in info]
            contents.append(info)
        if len(set([len(i) for i in contents])) > 1:
            print([len(i) for i in contents])
        return contents  # 表格质量很高了。

    def rank_the_rows(self, q_tokenized, tab_id):
        def flat_the_cells(cells):
            flatted_cell = []
            for cell in cells:
                flatted_cell.extend(cell)
            return flatted_cell

        row_score = defaultdict(int)
        table_rows = self.tables[tab_id]
        if args.use_summary_row == 'no':
            table_rows = table_rows[:-1]   # 这行很危险。。。
        else:
            row_score[len(table_rows)-1] += 3
        row_score[0] += 3
        for gram in q_tokenized:
            for row_id, row in enumerate(table_rows):
                row = flat_the_cells(row)
                if gram in row:
                    row_score[row_id] += 1
        row_ids_sorted = sorted(row_score, key=lambda rowid: row_score[rowid] * -1)[:5]
        # print("===" * 20)
        # print(u'|'.join(q_tokenized))
        # for rowid in row_ids_sorted[:10]:
        #     print(u'|'.join(flat_the_cells(table_rows[rowid])))
        return row_ids_sorted

    def get_dat(self, which):
        if which == 'train':
            fnm = 'data/train_id.json'
        elif which == 'val':
            fnm = 'data/val_id.json'
        elif which == 'test':
            fnm = 'data/test_id.json'
        elif which == 'simple':
            fnm = 'data/simple_test_id.json'
        elif which == 'complex':
            fnm = 'data/complex_test_id.json'
        else:
            assert 1 == 2, "which should be in train/val/test"
        table_ids = eval(open(fnm).read())
        smps = []
        for tab_id in tqdm(table_ids):
            tab_dat = self.all_dat[tab_id]
            qs, labels, caption = tab_dat
            for q, label in zip(qs, labels):
                q_tokenized = self.tokenizer.tokenize(q)
                kept_rows = self.rank_the_rows(q_tokenized, tab_id)
                smp = [q_tokenized, label, self.tokenizer.tokenize(caption), tab_id, kept_rows]
                smps.append(smp)
        print(f'{which} sample num={len(smps)}, pos_num={sum([i[1] for i in smps])}')
        return smps

    def load_all_the_dat(self):
        all_dat = {}
        for fnm in ["collected_data/r1_training_all.json", "collected_data/r2_training_all.json"]:
            infos = eval(open(fnm).read())
            all_dat.update(infos)
            print(len(all_dat))
        return all_dat

    def trans_a_smp(self, smp):
        q_tokenized, label, caption_tokenized, tab_id, kept_rows = smp
        table_contents = self.tables[tab_id]
        q_tokenized = ["[CLS]"] + q_tokenized + ['[SEP]'] + caption_tokenized + ["#"]
        type1_len = len(["[CLS]"] + q_tokenized + ['[SEP]'])
        type_ids = [0] * type1_len + [1] * (LL - type1_len)
        token_ids = self.tokenizer.convert_tokens_to_ids(q_tokenized)
        token_ids_source = [(-1, -1)] * len(q_tokenized)
        row_ids = []
        col_ids = []
        for colid in range(max([len(content) for content in table_contents])):
            for rowid, content in enumerate(table_contents):
                if rowid in kept_rows:
                    if len(content) > colid:
                        cell = content[colid]
                        token_ids.extend(self.tokenizer.convert_tokens_to_ids(cell))
                        token_ids_source.extend([(rowid, colid)] * len(cell))

#         for rowid, content in enumerate(table_contents):
#             if rowid in kept_rows:
#                 for cell_id, cell in enumerate(content):
#                     token_ids.extend(self.tokenizer.convert_tokens_to_ids(cell))
#                     token_ids_source.extend([(rowid, cell_id)] * len(cell))
        if len(token_ids) > LL:
            token_ids = token_ids[:LL]
            token_ids_source = token_ids_source[:LL]
        else:
            token_ids += [pad_token] * (LL - len(token_ids))
            token_ids_source += [(-2, -2)] * (LL - len(token_ids_source))
        # 可视化一下这个矩阵，确认是正确的才行。
        return token_ids, type_ids, token_ids_source, label  # build_vmm(token_ids_source)

    def trans2ids(self, smps):
        res = []
        for smp in tqdm(smps):
            res.append(self.trans_a_smp(smp))
        return res

    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            return DataLoader(shuffle=True, dataset=self.train_dataset, batch_size=10, num_workers=3)
        if which == 'dev':
            return DataLoader(shuffle=False, dataset=self.dev_dataset, batch_size=10, num_workers=3)
        if which == 'test':
            return DataLoader(shuffle=False, dataset=self.test_dataset, batch_size=10, num_workers=3)
        
        if which == 'simple':
            return DataLoader(shuffle=False, dataset=self.simple_smps, batch_size=10, num_workers=3)
        if which == 'complex':
            return DataLoader(shuffle=False, dataset=self.complex_smps, batch_size=10, num_workers=3)


class MyDataSet(Dataset):
    def __init__(self, smps):
        self.smps = smps
        super().__init__()

    def __getitem__(self, i):
        token_ids, type_ids, token_ids_source, label = self.smps[i]
        row_ids = [i for i,j in token_ids_source]
        col_ids = [j for i,j in token_ids_source]
        return np.asarray(token_ids), np.asarray(type_ids), np.asarray(build_vmm(token_ids_source, 'low')), np.asarray(build_vmm(token_ids_source, 'upper')), np.asarray(row_ids), np.asarray(col_ids),label
    def __len__(self):
        return len(self.smps)


def build_vmm(source_ids, low_or_upper):  # 主要耗时在这里。怎么加速下？
    vm = []
    for rowid, colid in source_ids:
        if rowid == -1 and colid == -1:
            vm.append([1] * LL)
            continue
        row_see = []
        for rowid2, colid2 in source_ids:
            if rowid2 == -1 and colid2 == -1:
                row_see.append(1)
            elif rowid2 == -2 and colid2 == -2:  # padding
                row_see.append(0)
            else:
                if rowid2 == 0:
                    row_see.append(1)
                else:
#                     if rowid == rowid2: #  or colid == colid2
                    if (low_or_upper == 'low' and rowid == rowid2) or (low_or_upper == 'upper' and (rowid == rowid2  or colid == colid2)):
                        row_see.append(1)
                    else:
                        row_see.append(0)
        assert len(row_see) == len(source_ids), "incorrect number"
        # if len(row_see) > LL:
        #     vm.append(row_see[:LL])
        # else:
        #     vm.append(row_see + [0] * (LL - len(row_see)))
        vm.append(row_see)
    return vm


# model and train and save and valid


class ModelDefine(nn.Module):
    def __init__(self):
        super(ModelDefine, self).__init__()
#         path = '/home/hadoop-aipnlp/cephfs/data/aikg-tag/freeze_bert'
        self.bert = RobertaModel.from_pretrained(roberta_dp)
        self.col_embs = nn.Embedding(1000, 768, padding_idx=0)
        self.row_embs = nn.Embedding(1000, 768, padding_idx=0)
        self.col_embs.weight.data = 0 * self.col_embs.weight.data
        self.col_embs.weight.data = 0 * self.col_embs.weight.data


#         self.bert = BertModel.from_pretrained(path)  # 定义一个模型
        if 'large' in roberta_dp:
            dim = 1024
        else:
            dim = 768
        self.classification_liner = nn.Linear(dim, 2)
        self.drop_out = nn.Dropout(0.1)

    #     def forward(self, w_idxs1, type_idxs, mask_idxs=None, vm=None):
    #         layer_outputs1, fts1 = self.bert(w_idxs1, type_idxs, attention_mask=w_idxs1 > 0.5)
    #         last_layer = layer_outputs1[-1]
    #         max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2), kernel_size=last_layer.size(1)).squeeze(-1)
    #         max_pooling_fts = self.drop_out(max_pooling_fts)
    #         return self.classification_liner(max_pooling_fts)

    def forward(self, w_idxs1, type_idxs, mask_idxs=None, vm_low=None, vm_upper=None,row_ids=None, col_ids=None):
        # layer_outputs1, fts1 = self.bert(w_idxs1, type_idxs, attention_mask=w_idxs1 > 0.5)

        embedding_output = self.bert.embeddings(w_idxs1, type_idxs * (0 if pad_token else 1))
        # 搞一个embedding出来 加到上面去
        col_embs = self.col_embs(col_ids+2)
        row_embs = self.row_embs(col_ids+2)
        embedding_output = embedding_output + row_embs + col_embs
        
        extended_attention_mask_base = mask_idxs.long().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = vm_low.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_low = (1.0 - extended_attention_mask) * -10000.0
        
        extended_attention_mask = vm_upper.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_upper = (1.0 - extended_attention_mask) * -10000.0
        
        extended_attention_mask = extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_baseline = (1.0 - extended_attention_mask) * -10000.0
        
        
#         #         extended_attention_mask = vm   感觉没这么简单阿
#         layer_outputs = self.bert.encoder(embedding_output,
#                                           extended_attention_mask, True)
        hidden_states = embedding_output
        for ly_idx, layer_module in enumerate(self.bert.encoder.layer):
            if ly_idx < 6:
                extended_attention_mask = extended_attention_mask_low
            else:
                extended_attention_mask = extended_attention_mask_upper                
            
#             hidden_states = layer_module(hidden_states)[0]
            if args.only_low_vm == 'yes':
                extended_attention_mask = extended_attention_mask_low

            if args.only_high_vm == 'yes':
                extended_attention_mask = extended_attention_mask_upper

            if args.use_vm == 'no':
                extended_attention_mask = extended_attention_mask_baseline
            hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
        last_layer = hidden_states

#         last_layer = layer_outputs[-1]
        max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2), kernel_size=last_layer.size(1)).squeeze(-1)
        max_pooling_fts = self.drop_out(max_pooling_fts)
        return self.classification_liner(max_pooling_fts)


class Model:
    def __init__(self, lr=2e-5, device=None):
        if device is None:
            self.device = 'cuda:1'
        else:
            self.device = device
        # self.device = 'cpu'

        self.model = ModelDefine()  # 定义一个模型
#         self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.updates = 0
        opt_layers = list(range(0, 12, 1))
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.dt = DataManager()
#     def trans_to_variable(self):
#         return 
        
    def train(self):
        for i in range(20):
            print("===" * 10)
            print("epoch%d" % i)
            for batch_id, batch in tqdm(enumerate(self.dt.iter_batches(which="train")), ncols=80):
                # trans to tensor
                batch_size = len(batch[0])
                self.model.train()
                word_idxs, type_idxs, vms_low, vms_upper, row_ids, col_ids, labels = [
                    Variable(e).long().to(self.device) for e
                    in batch]
                logits = self.model(word_idxs, type_idxs, word_idxs > 0.5, vms_low, vms_upper, row_ids, col_ids)  # pinyin_flags,
                loss = F.cross_entropy(logits, labels)
                self.train_loss.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 20)
                self.optimizer.step()
                self.updates += 1
                
                if batch_id == 3000:
                    print("middle: epoch {}, train loss={}".format(i, self.train_loss.avg))
                    self.validate(epoch=i, which='dev')
                    self.validate(epoch=i, which='test')
                
            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()
            self.validate(epoch=i, which='dev')
            self.validate(epoch=i, which='test')
#             self.validate(which="train", epoch=i)

    def validate(self, which="dev", epoch=-1):
        def simple_accuracy(preds1, labels1):
            correct_num = sum([1.0 if p1 == p2 else 0.0 for p1, p2 in zip(preds1, labels1)])
            return correct_num / len(preds1)

        preds_label = []
        gold_label = []
        y_predprob = []
        for batch in tqdm(self.dt.iter_batches(which=which), ncols=80):
            batch_size = len(batch[0])
            self.model.eval()
            word_idxs, type_idxs, vms_low, vms_upper,row_ids, col_ids, labels = [
                Variable(e).long().to(self.device) for e
                in batch]
            logits = self.model(word_idxs, type_idxs, word_idxs > 0.5, vms_low, vms_upper, row_ids, col_ids)  # pinyin_flags,
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            preds_label.extend(preds)
            gold_label.extend(batch[-1])
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, -1])
        preds_label = np.array(preds_label)
        gold_label = np.array(gold_label)
        acc = simple_accuracy(preds_label, gold_label)
        p, r, f1, _ = precision_recall_fscore_support(gold_label, preds_label)
        print("{} acc={}, p={}, r={}, f1={}".format(which, acc, p, r, f1))
        if which == "dev":
            self.save("models/mybertt4_new_{}_combine_cross_row_acc_{}.pt".format(epoch, acc), epoch)  # 跑一下模型。

    def batch_predict(self, test_tag_pairs, batch_size=None):
        samples = self.dt.trans_samples(test_tag_pairs)  # 这样做不如流式的处理，转一批、处理一批不占内存。# 有时间、有需要再修改
        y_predprob = []
        if len(samples) > 1000:
            batches = tqdm(self.dt.iter_batches(which="test", samples=samples, batch_size=batch_size))
        else:
            batches = self.dt.iter_batches(which="test", samples=samples)
        for batch in batches:
            self.model.eval()
            word_idxs, type_idxs, vms_low, vms_upper,row_ids, col_ids, labels = [
                Variable(e).long().to(self.device) for e
                in batch]
            logits = self.model(word_idxs, type_idxs, word_idxs > 0.5, vms_low, vms_upper,row_ids, col_ids)  # pinyin_flags,
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy())
        return y_predprob

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'epoch': epoch
        }
        torch.save(params, filename)

    def resume(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        state_dict = checkpoint['state_dict']

        # self.model = ModelDefine(state_dict=state_dict)  # 输入还没改过来

        new_state = set(self.model.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        self.model.load_state_dict(state_dict['network'])
        self.model.to(self.device)
        return self.model


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # DataManager()
    
#     t = DataManager()
#     for i in tqdm(t.iter_batches(which='train')):
#         print(i)
#         input()
    synonym_model = Model()
#     for i in range(8, 9, 1):
# #         print(i)
# #         synonym_model.resume("models/mybertt4_new_{}_combine_cross_row.pt".format(i))
#         synonym_model.resume("models/mybertt4_new_8_combine_cross_row_acc_0.5931055948953389.pt")
# #         synonym_model.validate(epoch=-1, which='dev')       
# #         synonym_model.validate(epoch=-1, which='test')
#         i = -1
#         synonym_model.validate(epoch=i, which='simple')
#         synonym_model.validate(epoch=i, which='complex')

#         synonym_model.validate(epoch=i, which='dev')
#         synonym_model.validate(epoch=i, which='test')

    #     synonym_model.resume("./models/gru_models_check_point_0.pt")
    synonym_model.train()
