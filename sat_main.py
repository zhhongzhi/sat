import os
import torch
import argparse
import random 
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--scan_direction',
                    type=str,
                    choices=["horizontal", "vertical"],
                    default='horizontal')

parser.add_argument('--use_vm',
                    type=str,
                    choices=["yes", "no"],
                    default='yes')

parser.add_argument('--reproduce',
                    type=str,
                    choices=["yes", "no"],
                    default='no')

parser.add_argument('--use_summary_row',
                    type=str,
                    choices=["yes", "no"],
                    default='yes')

parser.add_argument('--only_low_vm',
                    type=str,
                    choices=["yes", "no"],
                    default='no')

parser.add_argument('--only_high_vm',
                    type=str,
                    choices=["yes", "no"],
                    default='no')

args = parser.parse_args()
# args.use_vm = 'no'
# args.use_summary_row = 'no'


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


setup_seed(0)

from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np

import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

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

from transformers import BertModel, BertTokenizer

from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

LL = 256

bert_model_dp = 'bert-base-uncased'
pad_token = 0


class DataManager:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dp)

        import pickle
        tab_f = f"tables_and_count_info_{bert_model_dp[:-1]}.cache"
        if os.path.exists(tab_f):
            print('load tables from cache', tab_f)
            self.tables = pickle.load(open(tab_f, 'rb'))
        else:
            self.tables = self.load_the_tables()
            pickle.dump(self.tables, open(tab_f, 'wb'))

        self.all_dat = self.load_all_the_dat()
        train_smps_ = self.get_dat('train')
        test_smps_ = self.get_dat('test')
        valid_smps_ = self.get_dat('val')
        train_smps = self.trans2ids(train_smps_)
        test_smps = self.trans2ids(test_smps_)
        valid_smps = self.trans2ids(valid_smps_)

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
        def add_num_info(raw_table_contents, table_contents):
            from collections import defaultdict
            col_digit_flag = defaultdict(set)
            for content in raw_table_contents[1:]:
                for colid, col in enumerate(content):
                    col_digit_flag[colid].add(col.isdigit())
            digit_cols = []
            for colid, flags in col_digit_flag.items():
                if len(flags) == 1 and True in flags:
                    digit_cols.append(colid)

            for colid in digit_cols:
                col_infos = [content[colid] for content in raw_table_contents]
                idxs = sorted(range(1, len(col_infos)), key=lambda idx: float(col_infos[idx]) * -1)
                for rank_num, idx in enumerate(idxs):
                    table_contents[idx][colid].append("[unused{}]".format(rank_num))
            return table_contents

        def add_count_row(raw_table_contents, table_contents):
            count_row = []
            for colid in range(len(raw_table_contents[0])):
                col_cells = [row[colid] for row in raw_table_contents]
                if len(col_cells) == len(set(col_cells)):
                    count_row.append([])
                else:
                    cell_cnt = defaultdict(int)
                    for cell in col_cells:
                        cell_cnt[cell] += 1
                    cells = sorted(cell_cnt, key=lambda cell: cell_cnt[cell] * -1)
                    cell_cnt_l = [f'{cell} {cell_cnt[cell]}' for cell in cells if cell_cnt[cell] > 1]
                    count_row.append(u'count :' + u', '.join(cell_cnt_l))
            table_contents.append([self.tokenizer.tokenize(cell) if len(cell) else [] for cell in count_row])
            return table_contents

        contents = []
        raw_contents = []
        for ln in open(fnm).readlines():
            info = ln.strip("\n").split("#")
            # print(info)
            raw_contents.append(info)
            info = [self.tokenizer.tokenize(i) for i in info]
            # print(info)
            contents.append(info)
        contents = add_count_row(raw_contents, contents)
        if len(set([len(i) for i in contents])) > 1:
            print([len(i) for i in contents])
        return contents  
    
    def rank_the_rows(self, q_tokenized, tab_id):
        def flat_the_cells(cells):
            flatted_cell = []
            for cell in cells:
                flatted_cell.extend(cell)
            return flatted_cell

        row_score = defaultdict(int)
        table_rows = self.tables[tab_id]
        if args.use_summary_row == 'no':
            table_rows = table_rows[:-1]
        else:
            row_score[len(table_rows) - 1] += 3
        row_score[0] += 3
        for gram in q_tokenized:
            for row_id, row in enumerate(table_rows):
                row = flat_the_cells(row)
                if gram in row:
                    row_score[row_id] += 1
        row_ids_sorted = sorted(row_score, key=lambda rowid: row_score[rowid] * -1)[:5]
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
        if args.scan_direction == 'vertical':
            for colid in range(max([len(content) for content in table_contents])):
                for rowid, content in enumerate(table_contents):
                    if rowid in kept_rows:
                        if len(content) > colid:
                            cell = content[colid]
                            token_ids.extend(self.tokenizer.convert_tokens_to_ids(cell))
                            token_ids_source.extend([(rowid, colid)] * len(cell))
        elif args.scan_direction == 'horizontal':
            for rowid, content in enumerate(table_contents):
                if rowid in kept_rows:
                    for cell_id, cell in enumerate(content):
                        token_ids.extend(self.tokenizer.convert_tokens_to_ids(cell))
                        token_ids_source.extend([(rowid, cell_id)] * len(cell))
        else:
            assert 1 == 2, "not support scan direction"
        if len(token_ids) > LL:
            token_ids = token_ids[:LL]
            token_ids_source = token_ids_source[:LL]
        else:
            token_ids += [pad_token] * (LL - len(token_ids))
            token_ids_source += [(-2, -2)] * (LL - len(token_ids_source))
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
        return np.asarray(token_ids), np.asarray(type_ids), np.asarray(build_vmm(token_ids_source, 'low')), np.asarray(
            build_vmm(token_ids_source, 'upper')), label

    def __len__(self):
        return len(self.smps)


def build_vmm(source_ids, low_or_upper):
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
                    if (low_or_upper == 'low' and rowid == rowid2) or (
                            low_or_upper == 'upper' and (rowid == rowid2 or colid == colid2)):
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



class ModelDefine(nn.Module):
    def __init__(self):
        super(ModelDefine, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_dp)
        dim = 768
        self.classification_liner = nn.Linear(dim, 2)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, w_idxs1, type_idxs, mask_idxs=None, vm_low=None, vm_upper=None):

        embedding_output = self.bert.embeddings(w_idxs1, type_idxs * (0 if pad_token else 1))

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

        hidden_states = embedding_output
        for ly_idx, layer_module in enumerate(self.bert.encoder.layer):
            if ly_idx < 6:
                extended_attention_mask = extended_attention_mask_low
            else:
                extended_attention_mask = extended_attention_mask_upper

            if args.only_low_vm == 'yes':
                extended_attention_mask = extended_attention_mask_low

            if args.only_high_vm == 'yes':
                extended_attention_mask = extended_attention_mask_upper

            if args.use_vm == 'no':
                extended_attention_mask = extended_attention_mask_baseline
            hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
        last_layer = hidden_states

        max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2), kernel_size=last_layer.size(1)).squeeze(-1)
        max_pooling_fts = self.drop_out(max_pooling_fts)
        return self.classification_liner(max_pooling_fts)


class Model:
    def __init__(self, lr=2e-5, device=None):
        if device is None:
            self.device = 'cuda:0'
        else:
            self.device = device

        self.model = ModelDefine()
        #         self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.updates = 0
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.dt = DataManager()

    def train(self):
        for i in range(20):
            print("===" * 10)
            print("epoch%d" % i)
            for batch_id, batch in tqdm(enumerate(self.dt.iter_batches(which="train")), ncols=80):
                batch_size = len(batch[0])
                self.model.train()
                word_idxs, type_idxs, vms_low, vms_upper, labels = [
                    Variable(e).long().to(self.device) for e
                    in batch]
                logits = self.model(word_idxs, type_idxs, word_idxs > 0.5, vms_low, vms_upper) 
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
                    self.validate(epoch=i, which='simple')
                    self.validate(epoch=i, which='complex')

            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()
            self.validate(epoch=i, which='dev')
            self.validate(epoch=i, which='test')
            self.validate(epoch=i, which='simple')
            self.validate(epoch=i, which='complex')

    def validate(self, which="dev", epoch=-1):
        predicted_labels_lns = []
        def simple_accuracy(preds1, labels1):
            correct_num = sum([1.0 if p1 == p2 else 0.0 for p1, p2 in zip(preds1, labels1)])
            return correct_num / len(preds1)

        preds_label = []
        gold_label = []
        y_predprob = []
        for batch in tqdm(self.dt.iter_batches(which=which), ncols=80):
            batch_size = len(batch[0])
            self.model.eval()
            word_idxs, type_idxs, vms_low, vms_upper, labels = [
                Variable(e).long().to(self.device) for e
                in batch]
            logits = self.model(word_idxs, type_idxs, word_idxs > 0.5, vms_low, vms_upper)  # pinyin_flags,
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            preds_label.extend(preds)
            gold_label.extend(batch[-1])
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, -1])
        for p_label, g_label, s in zip(preds_label, gold_label, y_predprob):
            predicted_labels_lns.append(u'{}\t{}\t{}\t{}\n'.format(p_label, g_label, s, 'error' if g_label != p_label else 'correct'))
        open('prediction_result_of_{}.txt'.format(which), "w").writelines(predicted_labels_lns)
        preds_label = np.array(preds_label)
        gold_label = np.array(gold_label)
        acc = simple_accuracy(preds_label, gold_label)
        p, r, f1, _ = precision_recall_fscore_support(gold_label, preds_label)
        print("{} acc={}, p={}, r={}, f1={}".format(which, acc, p, r, f1))
        if which == "dev" and epoch != -1:
            self.save("epoch_{}_acc_{}.pt".format(epoch, acc), epoch)  # 跑一下模型。

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
    if args.reproduce == 'yes':
        model = Model()
        args = parser.parse_args()
        args.use_vm = 'yes'
        args.use_summary_row = 'yes'
        print(args) 
        model_path = "freezed_model/epoch_15_acc_0.7330947008014941.pt"
        print(model_path)
        model.resume(model_path)
        model.validate(epoch=-1, which='simple')
        model.validate(epoch=-1, which='complex')
        model.validate(epoch=-1, which='dev')
        model.validate(epoch=-1, which='test')
    else:
        model = Model()
        model.train()
        
