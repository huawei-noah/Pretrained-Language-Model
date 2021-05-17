#-*- coding:utf-8 -*-
#The MIT License (MIT)
#Copyright (c) 2021 Huawei Technologies Co., Ltd.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif,build_dataloader
from radam_optimizer import *


def train(config, model, train_data, dev_data, test_data, eval_per_batchs=100):
    start_time = time.time()
    model.train()
    optimizer = RiemannianAdam(model.parameters(), lr=config.learning_rate)

    total_batch = 0
    dev_best_acc = 0.0
    last_improve = 0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:{}/{}'.format(epoch + 1, config.num_epochs))
        train_data.initial()
        train_iter = build_dataloader(train_data, config.batch_size,True)
        for i, (trains, labels) in enumerate(train_iter):
            trains1=trains[0].to(config.device)
            trains2 = trains[1].to(config.device)

            trains=(trains1,trains2)
            labels = labels.to(config.device)

            outputs = model(trains)
            model.zero_grad()

            labels = torch.squeeze(labels)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model.embedding.data[0] = nn.init.zeros_(model.embedding[0])
                model.embedding_wordngram.data[0] = nn.init.zeros_(model.embedding_wordngram[0])

            if total_batch % eval_per_batchs == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_data)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '**'
                    last_improve = total_batch
                else:
                    improve = ''
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * config.lr_decay_rate
                time_dif = get_time_dif(start_time)

                msg = 'Train Step: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.4%},  Val Loss: {3:>5.4},  Val Acc: {4:>6.4%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No improving for a long time, auto early stopping...")
                flag = True
                break

        if flag:
            break
    test(config, model, test_data)

def test(config, model, test_data):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_data, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Test Time:", time_dif)


def evaluate(config, model, dev_data, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    dev_data.initial()
    dev_iter = build_dataloader(dev_data, config.batch_size,False)
    with torch.no_grad():
        for (evals, labels) in dev_iter:
            evals1 = evals[0].to(config.device)
            evals2 = evals[1].to(config.device)

            evals = (evals1, evals2)
            labels = labels.to(config.device)

            outputs = model(evals)
            labels = torch.squeeze(labels)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / (len(dev_data)/config.batch_size), report, confusion
    return acc, loss_total / (len(dev_data)/config.batch_size)
