from collections import defaultdict
import copy
from sklearn.decomposition import PCA, FastICA
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFedPT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.local_proto = None
        self.global_proto = None

        self.MSE_loss = nn.MSELoss()

        self.label_size = torch.zeros(self.num_classes)
        self.feature_dim = args.feature_dim

        self.fine_tuning_epochs = args.fine_tuning_epochs

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.label_size = torch.zeros(self.num_classes, device=self.device, dtype=torch.int)
        for i, (x, y) in enumerate(trainloader):
            y = y.to(self.device)  # 将当前批次数据转移到指定设备
            self.label_size += torch.bincount(y.flatten(), minlength=self.num_classes)
        # print(self.label_size)

        self.model.train()

        # for param in self.model.base.parameters():
        #     param.requires_grad = True
        # for param in self.model.head.parameters():
        #     param.requires_grad = False
        # for epoch in range(self.local_epochs):
        #     for i, (x, y) in enumerate(trainloader):
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         output = self.model(x)
        #         loss = self.loss(output, y)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        # for param in self.model.base.parameters():
        #     param.requires_grad = False
        # for param in self.model.head.parameters():
        #     param.requires_grad = True
        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                if self.global_proto is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for j, yy in enumerate(y):
                        proto_new[j, :] = self.global_proto[yy.item()]
                    loss += self.loss(self.model.head(proto_new), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_protos(self, global_proto):
        self.global_proto = global_proto

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        self.local_proto = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                for j, yy in enumerate(y):
                    self.local_proto[yy.item()] += rep[j, :]
        label_size_safe = self.label_size.clamp(min=1).unsqueeze(1)  # 防止除以零，当某个类别的样本数为0时，保持原值不变
        self.local_proto /= label_size_safe  # 对 local_proto 进行行归一化，使其每个类别的原型向量表示该类所有样本特征向量的平均值

    def set_base_model(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def set_head_model(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def fine_tune(self, which_module=['base', 'head']):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()

        if 'head' in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = True

        if 'base' not in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = False

        for _ in range(self.fine_tuning_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) is type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['total_cost'] += time.time() - start_time
