import copy
import random
import time
from collections import defaultdict
import itertools
import statistics

import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientFedPT import clientFedPT
from flcore.servers.serverbase import Server
from scipy.optimize import linprog
from torch.utils.data import DataLoader


class FedPT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.num_clients = args.num_clients

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedPT)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.feature_dim = args.feature_dim
        self.global_epochs = args.global_epochs
        self.fine_tuning_epochs = args.fine_tuning_epochs

        self.global_head_model = Global_Head(args.num_classes, args.feature_dim).to(self.device)
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.KL_loss = nn.KLDivLoss()
        
        self.optimizer = torch.optim.SGD(self.global_head_model.parameters(), lr=args.global_learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay

        self.alpha = args.alpha   # 权重系数
        print(f"alpha: {self.alpha}")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i == 0:
                self.send_model()
            else:
                self.send_base_and_head_model()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("Evaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            active_clients = random.sample(self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))
            self.receive_models_and_protos(active_clients)
            self.aggregate_base_parameters()
            self.aggregate_head_parameters()

            self.proto_aggregation()
            self.send_protos()
            print("Global_train")
            self.train_model(active_clients)

            self.budget.append(time.time() - s_t)
            print('-' * 25, 'time cost:', self.budget[-1], '-' * 25)

        print("\nBest accuracy: ", max(self.rs_test_acc))
        print("\nAverage time cost per round: ", sum(self.budget[1:]) / len(self.budget[1:]))

        for client in self.clients:
            client.fine_tune()
        print("\n-------------Evaluate fine-tuned personalized models-------------")
        self.evaluate()

        self.save_results()
        self.save_global_model()

    def send_base_and_head_model(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_base_model(self.global_base_model)
            client.set_head_model(self.global_head_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models_and_protos(self, active_clients):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_label_size = []
        # self.uploaded_label_distribution = []
        self.uploaded_dataset_size = []
        self.uploaded_matrix = []
        self.uploaded_reduced_local_protos = []
        self.uploaded_base_models = []
        self.uploaded_head_models = []
        self.uploaded_models = []

        self.uploaded_local_protos_x = []
        self.uploaded_local_protos_y = []

        self.total_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.total_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_label_size.append(client.label_size)
                # self.uploaded_label_distribution.append([element / sum(client.label_size) for element in client.label_size])
                self.uploaded_dataset_size.append(client.train_samples)

                self.uploaded_base_models.append(client.model.base)
                self.uploaded_head_models.append(client.model.head)
                self.uploaded_models.append(client.model)

                self.uploaded_local_protos_x.append(client.local_proto)
                y_tensor = torch.arange(0, self.num_classes, device=self.device, dtype=torch.int64)
                self.uploaded_local_protos_y.append(y_tensor)

        self.uploaded_weights = [element / self.total_samples for element in self.uploaded_dataset_size]

        self.uploaded_label_size = torch.stack(self.uploaded_label_size).to(self.device)
        # self.uploaded_label_size = torch.where(self.uploaded_label_size > 0, torch.ones_like(self.uploaded_label_size), self.uploaded_label_size)
        column_sums = self.uploaded_label_size.sum(dim=0)  # 计算每列的元素之和
        self.uploaded_label_distribution = (self.uploaded_label_size / column_sums.clamp(min=1)).to(self.device)

    def aggregate_base_parameters(self):
        assert (len(self.uploaded_base_models) > 0)
        self.global_base_model = copy.deepcopy(self.uploaded_base_models[0])
        for param in self.global_base_model.parameters():
            param.data.zero_()
        for w, client_base_model in zip(self.uploaded_weights, self.uploaded_base_models):
            for server_param, client_param in zip(self.global_base_model.parameters(), client_base_model.parameters()):
                server_param.data += client_param.data.clone() * w

    def aggregate_head_parameters(self):
        assert (len(self.uploaded_head_models) > 0)
        for param in self.global_head_model.parameters():
            param.data.zero_()
        for w, client_head_model in zip(self.uploaded_weights, self.uploaded_head_models):
            for server_param, client_param in zip(self.global_head_model.parameters(), client_head_model.parameters()):
                server_param.data += client_param.data.clone() * w

    def proto_aggregation(self):
        # uploaded_local_protos = torch.stack(self.uploaded_local_protos_x).to(self.device)
        # global_proto = uploaded_local_protos * self.uploaded_label_distribution.unsqueeze(-1).expand(-1, -1, 512)
        # global_proto = global_proto.sum(dim=0).to(self.device)

        uploaded_local_protos_x = torch.cat(self.uploaded_local_protos_x, dim=0)
        uploaded_local_protos_y = torch.cat(self.uploaded_local_protos_y, dim=0)

        non_zero_rows = uploaded_local_protos_x.any(dim=1)
        uploaded_local_protos_x = uploaded_local_protos_x[non_zero_rows]
        uploaded_local_protos_y = uploaded_local_protos_y[non_zero_rows]
        self.train_proto = [(rep, y) for rep, y in zip(uploaded_local_protos_x, uploaded_local_protos_y)]

        grouped_reps = defaultdict(list)
        for rep, y in self.train_proto:
            grouped_reps[y.item()].append(rep)

        self.global_proto = torch.zeros((self.num_classes, self.feature_dim), dtype=torch.float32, device=self.device)
        self.global_proto_std = torch.zeros((self.num_classes, self.feature_dim), dtype=torch.float32, device=self.device)
        for label, reps in grouped_reps.items():
            stacked_reps = torch.stack(reps, dim=0)
            mean = stacked_reps.mean(dim=0)
            std = stacked_reps.std(dim=0)
            self.global_proto[label] = mean
            self.global_proto_std[label] = std

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_proto)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def train_model(self, active_clients):
        trainload = DataLoader(self.train_proto, batch_size=10, drop_last=True, shuffle=True)

        for epoch in range(self.global_epochs):
            total = 0
            correct = 0
            self.global_head_model.train()
            for param in self.global_head_model.parameters():
                param.requires_grad = True
            for j, (rep, y) in enumerate(trainload):
                for j, yy in enumerate(y):
                    rep[j, :] = self.global_proto[yy.item()] + 1.5 * torch.randn_like(rep[j, :]) * self.global_proto_std[yy.item()]
                rep = rep.to(self.device)
                y = y.to(self.device)
                output = self.global_head_model(rep)
                loss = self.alpha * self.CE_loss(output, y)

                proto_new = copy.deepcopy(rep.detach())
                label_distribution = torch.zeros((y.shape[0], len(active_clients)), dtype=torch.float32)
                for k, yy in enumerate(y):
                    # proto_new[k, :] = self.global_proto[yy.item()]
                    label_distribution[k] = self.uploaded_label_distribution.T[yy.item()]
                label_distribution = label_distribution.T
                logits = 0
                for w, model in zip(label_distribution, self.uploaded_head_models):
                    w = torch.Tensor(w).type(torch.float32)
                    model.eval()
                    y_prediction = model(proto_new)
                    for ww, data in zip(w, y_prediction.detach()):
                        data *= ww
                    logits += y_prediction
                loss += (1 - self.alpha) * self.KL_loss(nn.functional.log_softmax(output, dim=1), nn.functional.softmax(logits, dim=1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predicted = torch.argmax(output, dim=1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = correct / total
            print(f"epoch {epoch} train head model acc: {accuracy}")


class Global_Head(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, rep):
        out = self.fc(rep)
        return out