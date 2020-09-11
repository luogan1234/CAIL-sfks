import json
import os
import tqdm
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_pretrained_bert import BertTokenizer

class SFKSDataset(Dataset):
    def __init__(self, path, task, train):
        super().__init__()
        self.task = task
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese-vocab.txt')
        raw = []
        files = os.listdir(path)
        for file in files:
            with open(path+file, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    l = len(obj.get('answer', []))
                    if train and (task == 1 and l > 1 or task == 2 and l == 1):
                        continue
                    raw.append(obj)
        random.shuffle(raw)
        self.data = raw
    
    def get_trues(self):
        res = {}
        for datum in self.data:
            res[datum['id']] = datum.get('answer', [])
        return res
    
    def convert(self, text):
        tokens = self.tokenizer.tokenize(text)[:512]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        type = int(datum['type'])
        #subject = datum.get('subject', '')
        query = self.convert(datum['statement'])
        options = [self.convert(datum['option_list'][key]) for key in ['A', 'B', 'C', 'D']]
        ans = datum.get('answer', [])
        if self.task == 0:
            label = [len(ans)>1]
        if self.task == 1:
            label = ord(ans[0])-ord('A') if ans else 0
        if self.task == 2:
            label = ['A' in ans, 'B' in ans, 'C' in ans, 'D' in ans]
        item = {'id': datum['id'], 'query': query, 'options': options, 'label': label, 'type': type}
        return item

class MyBatch:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, data):
        ids, types, query_inputs, option_inputs, labels = [], [], [], [], []
        for datum in data:
            ids.append(datum['id'])
            types.append(datum['type'])
            query_inputs.append(datum['query'])
            option_inputs.extend(datum['options'])
            labels.append(datum['label'])
        query_inputs = self.padding(query_inputs)
        option_inputs = self.padding(option_inputs)
        query_inputs = torch.tensor(query_inputs, dtype=torch.long).cuda()
        option_inputs = torch.tensor(option_inputs, dtype=torch.long).cuda()
        obj = {'ids': ids, 'labels': labels, 'types': types, 'query_inputs': query_inputs, 'option_inputs': option_inputs}
        return obj
        
    def padding(self, arr):
        n = min(self.config.max_len, max([len(a) for a in arr]))
        arr = [([101]+a[:n-2]+[102]+[0]*(n-len(a[:n-2])-2)) for a in arr]
        return arr

class SKFSDataLoader:
    def __init__(self, config, path):
        super().__init__()
        self.datasets = [SFKSDataset(path, task, config.train) for task in range(3)]
        self.batch_size = config.batch_size
        self.fn = MyBatch(config)
    
    def get_train(self, task):
        n = len(self.datasets[task])
        d1, d2 = random_split(self.datasets[task], [int(n*0.9), n-int(n*0.9)])
        train = DataLoader(d1, self.batch_size, shuffle=True, collate_fn=self.fn)
        eval = DataLoader(d2, self.batch_size, shuffle=True, collate_fn=self.fn)
        return train, eval
    
    def get_test(self, task):
        test = DataLoader(self.datasets[task], self.batch_size, shuffle=False, collate_fn=self.fn)
        return test