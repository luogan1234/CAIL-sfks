import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.model_classify import ModelClassify
from model.model_pred import ModelPred

class Processor:
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
    
    def ce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def bce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.float).cuda()
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        return loss
    
    def train_one_step(self, data, loss_fn):
        outputs = self.model(data['query_inputs'], data['option_inputs'], data['types'])
        loss = loss_fn(outputs, data['labels'])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def eval_one_step(self, data, loss_fn):
        with torch.no_grad():
            outputs = self.model(data['query_inputs'], data['option_inputs'], data['types'])
            loss = loss_fn(outputs, data['labels'])
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()  # sigmoid不影响argmax
        return outputs, loss.item()
    
    def get_res(self, ids, preds):
        res = {}
        for i, id in enumerate(ids):
            res[id] = preds[i]
        return res
    
    def evaluate(self, data, task):
        self.model.eval()
        ids, trues, preds = [], [], []
        eval_loss = 0
        eval_tqdm = tqdm.tqdm(enumerate(data))
        eval_tqdm.set_description("eval_loss: %.3f" % (0.0))
        for i, batch_data in eval_tqdm:
            ids.extend(batch_data['ids'])
            trues.extend(batch_data['labels'])
            if task == 1:
                outputs, loss = self.eval_one_step(batch_data, self.ce_loss)
                outputs = np.argmax(outputs, axis=1).tolist()
            else:
                outputs, loss = self.eval_one_step(batch_data, self.bce_loss)
                outputs = (outputs>0.5).tolist()
            preds.extend(outputs)
            eval_loss += loss
            eval_tqdm.set_description("eval_loss: %.3f" % (loss))
        eval_loss /= len(data)
        self.model.train()
        res = self.get_res(ids, preds)
        acc = accuracy_score(trues, preds)
        print('average eval_loss: {:.3f}, acc: {:.3f}'.format(eval_loss, acc))
        if task == 0:
            p = precision_score(1-np.array(trues), 1-np.array(preds))
            r = recall_score(1-np.array(trues), 1-np.array(preds))
            f1 = f1_score(1-np.array(trues), 1-np.array(preds))
            acc = f1
            print('precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, predict single number: {}'.format(p, r, f1, len(preds)-np.sum(preds)))
        return acc, res
    
    def train(self):
        print('Train starts.')
        for task in range(3):
            if task == 0:
                self.model = ModelClassify(self.config)
            else:
                self.model = ModelPred(self.config)
            print('model parameters number: {}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
            self.model.cuda()
            best_para = self.model.state_dict()
            train, eval = self.data_loader.get_train(task)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            print('Batch data number: train {}, eval {}'.format(len(train), len(eval)))
            best_acc, best_res = 0.0, []
            patience = 0
            try:
                for epoch in range(self.config.max_epochs):
                    train_loss = 0.0
                    train_tqdm = tqdm.tqdm(enumerate(train))
                    train_tqdm.set_description("Epoch %d | train_loss: %.3f" % (epoch, 0.0))
                    for i, batch_data in train_tqdm:
                        loss = self.train_one_step(batch_data, self.ce_loss if task == 1 else self.bce_loss)
                        train_loss += loss
                        train_tqdm.set_description("Epoch %d | train_loss: %.3f" % (epoch, loss))
                    print('average train_loss: {:.3f}'.format(train_loss / len(train)))
                    if epoch >= self.config.min_check_epoch:
                        acc, res = self.evaluate(eval, task)
                        if acc > best_acc:
                            patience = 0
                            best_para = self.model.state_dict()
                            best_acc = acc
                            best_res = res
                        patience += 1
                        if patience > self.config.early_stop_time:
                            train_tqdm.close()
                            break
            except KeyboardInterrupt:
                train_tqdm.close()
                print('Exiting from training early, stop at epoch {}'.format(epoch))
            print('Train finished, stop at {} epochs, best acc or f1 {:.3f}'.format(epoch, best_acc))
            with open('result/model_states/{}.pth'.format(self.config.store_name(task)), 'wb') as f:
                torch.save(best_para, f)
            with open('result/result.txt', 'a', encoding='utf-8') as f:
                obj = self.config.parameter_info(task)
                obj.update({'acc or f1': round(best_acc, 3)})
                f.write(json.dumps(obj)+'\n')
            with open('result/predict_{}.json'.format(task), 'w', encoding='utf-8') as f:
                json.dump(best_res, f)
    
    def test(self, res_path):
        print('Test starts.')
        res = []
        for task in range(3):
            if task == 0:
                self.model = ModelClassify(self.config)
            else:
                self.model = ModelPred(self.config)
            print('model parameters number: {}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
            self.model.cuda()
            with open('result/model_states/{}.pth'.format(self.config.store_name(task)), 'rb') as f:
                best_para = torch.load(f)
            self.model.load_state_dict(best_para)
            self.model.eval()
            data = self.data_loader.get_test(task)
            ids, preds = [], []
            for i, batch_data in tqdm.tqdm(enumerate(data)):
                ids.extend(batch_data['ids'])
                if task == 1:
                    outputs, loss = self.eval_one_step(batch_data, self.ce_loss)
                    outputs = np.argmax(outputs, axis=1).tolist()
                else:
                    outputs, loss = self.eval_one_step(batch_data, self.bce_loss)
                    outputs = (outputs>0.5).tolist()
                preds.extend(outputs)
            res.append(self.get_res(ids, preds))
        final = {}
        for key in res[0]:
            final[key] = []
            if res[0][key][0]:
                for j in range(4):
                    if res[2][key][j]:
                        final[key].append(chr(ord('A')+j))
            else:
                final[key].append(chr(ord('A')+res[1][key]))
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(final, f)