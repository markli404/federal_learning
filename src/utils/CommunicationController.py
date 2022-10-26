import numpy as np
import logging
import copy
import gc
import operator
# custom packages
from ..config import config
from ..utils.Printer import *
import torch
from collections import OrderedDict
import time

class CommunicationController:
    def __init__(self, clients):
        self.weight = None
        self.improvement = None
        self.num_clients = len(clients)
        self.clients = clients
        self.sampled_clients_indices = None

        # FedPNS
        self.test_count = np.zeros(len(clients))

    def update_weight(self):
        if self.weight is None:
            self.weight = np.ones(self.num_clients) / self.num_clients

        weight = []
        for client in self.clients:
            # if self.sampled_clients_indices is not None and client.id in self.sampled_clients_indices and self.improvement is not None:
            #     weight.append(max(0.001, self.improvement[client.id] - 0.15))
            # else:
            #     weight.append(client.get_performance_gap())

            weight.append(client.get_performance_gap())

        # self.improvement = np.array(weight)
        # self.weight = np.array(weight) / sum(weight)
        self.weight = np.minimum(self.weight + np.array(weight) - np.ones(self.num_clients) * config.DECAY, np.ones(len(weight)))
        self.weight = np.maximum(self.weight, np.ones(len(weight)) * config.BASE)

        self.improvement = self.weight

        message = f"Current clients have weights: {pretty_list(self.weight)} and have improvement: {pretty_list(weight)}"
        return message

    def sample_clients_test(self):
        if self.improvement is None:
            return self.sample_clients()

        frequency = self.weight
        # for improvement in self.improvement:
        #     frequency.append(1/(1 + np.exp(-config.C_1 * (improvement - config.C_2))))

        random_numbers = np.random.uniform(0, 1, len(frequency))

        sampled_client_indices = [idx for idx, val in enumerate(frequency) if val >= random_numbers[idx]]
        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update with possibility {np.array(frequency)[sampled_client_indices]}."

        return message, sampled_client_indices


    def sample_clients_random(self):
        num_sampled_clients = max(int(config.FRACTION * self.num_clients), 1)
        sampled_client_indices = np.random.choice(self.num_clients, num_sampled_clients, replace=False).tolist()

        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update."

        return message, sampled_client_indices


    def sample_clients_casestudy(self):
        frequency = []
        num_sampled_clients = max(int(config.FRACTION * self.num_clients), 1)
        for client in self.clients:
            if client.temporal_heterogeneous:
                frequency.append(5)
            else:
                frequency.append(0.0000001)

        frequency = np.array(frequency) / sum(frequency)
        sampled_client_indices = np.random.choice(self.num_clients, num_sampled_clients, p=frequency, replace=False).tolist()
        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update."

        return message, sampled_client_indices


    def sample_clients_fed_pns(self, clients, test_set):
        def average(grad_all):

            value_list = list(grad_all.values())

            w_avg = copy.deepcopy(value_list[0])
            # print(type(w_avg))
            for i in range(1, len(value_list)):
                w_avg += value_list[i]
            return w_avg / len(value_list)

        def client_deleting(expect_list, expect_value, selected_clients, local_gradients):
            for i in range(len(selected_clients)):
                worker_ind_del = [n for n in selected_clients if n != selected_clients[i]]
                grad_del = local_gradients.copy()
                grad_del.pop(selected_clients[i])
                avg_grad_del = average(grad_del)
                grad_del['avg_grad'] = avg_grad_del
                expect_value_del = get_relation(grad_del, worker_ind_del)
                expect_list[selected_clients[i]] = expect_value_del
            expect_list['all'] = expect_value

            return expect_list

        def get_relation(avg_grad, idxs_users):
            def dot_sum(K, L):
                return round(sum(i[0] * i[1] for i in zip(K.numpy(), L.numpy())), 2)

            innnr_value = {}

            for i in range(len(idxs_users)):
                innnr_value[idxs_users[i]] = dot_sum(avg_grad[idxs_users[i]], avg_grad['avg_grad'])

            return round(sum(list(innnr_value.values())), 3)

        def test_part(clients, selected_clients, test_set, key):
            model = FedAvg(clients, selected_clients)
            _, loss_all = model_evaluation_simple(model, test_set)

            selected_clients.remove(key)
            model = FedAvg(clients, selected_clients)
            _, loss_part = model_evaluation_simple(model, test_set)
            return loss_all, loss_part

        def model_evaluation_simple(model, test_set):
            # calculate the sample distribution of all clients
            device = config.DEVICE
            model.eval()
            model.to(device)

            test_loss, correct = 0, 0
            with torch.no_grad():
                for data, labels in test_set.get_dataloader():
                    data, labels = data.float().to(device), labels.long().to(device)
                    outputs = model(data)
                    test_loss += eval(config.CRITERION)()(outputs, labels).item()

                    predicted = outputs.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(labels.view_as(predicted)).sum().item()

                    if device == "cuda": torch.cuda.empty_cache()
            model.to("cpu")

            # calculate the metrics
            test_loss = test_loss / len(test_set.get_dataloader())
            test_accuracy = correct / len(test_set)
            return test_accuracy, test_loss

        def FedAvg(clients, selected_clients):
            fedavg_coeff = [len(clients[idx]) for idx in selected_clients]
            fedavg_coeff = np.array(fedavg_coeff) / sum(fedavg_coeff)

            new_model = copy.deepcopy(clients[0].model)
            averaged_weights = OrderedDict()

            for it, idx in enumerate(selected_clients):
                local_weights = clients[idx].model.state_dict()
                for key in new_model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = fedavg_coeff[it] * local_weights[key]
                    else:
                        averaged_weights[key] += fedavg_coeff[it] * local_weights[key]

            new_model.load_state_dict(averaged_weights)
            return new_model

        selected_clients = list(range(self.num_clients))

        st = time.time()

        local_gradients = {}
        for client in self.clients:
            local_gradients[client.id] = client.get_gradient()
        local_gradients['avg_grad'] = average(local_gradients)
        max_now = get_relation(local_gradients, selected_clients)
        local_gradients.pop('avg_grad')

        et = time.time()
        elapsed_time = et - st
        print('Get gradients:', elapsed_time, 'seconds')

        expect_list = {}
        labeled = []
        st = time.time()

        num_sampled_clients = max(int(config.FRACTION * self.num_clients), 1)
        while len(selected_clients) > num_sampled_clients:
            expect_list = client_deleting(expect_list, max_now, selected_clients, local_gradients)
            # print(len(w_locals), expect_list)
            key = max(expect_list.items(), key=operator.itemgetter(1))[0]
            if expect_list[key] <= expect_list["all"]:
                break
            else:
                labeled.append(key)
                # TODO self.test_count[key][1] += 1
                expect_list.pop("all")
                loss_all, loss_pop = test_part(clients, selected_clients, test_set, key)

                if loss_all < loss_pop:
                    selected_clients.append(key)
                    break
                else:
                    local_gradients.pop(key)
                    max_now = expect_list[key]
                    expect_list.pop(key)
        et = time.time()
        elapsed_time = et - st
        print('Select Samples:', elapsed_time, 'seconds')
        self.sampled_clients_indices = selected_clients
        message = f"{selected_clients} clients are selected for the next update."

        return message, selected_clients

    def sample_clients(self):
        if self.weight is None:
            self.weight = np.ones(len(self.clients)) / len(self.clients)

        p = np.array(self.weight) / sum(self.weight)
        num_sampled_clients = max(int(config.FRACTION * self.num_clients), 1)
        client_indices = [i for i in range(self.num_clients)]
        sampled_client_indices = sorted(
            np.random.choice(a=client_indices, size=num_sampled_clients, replace=False, p=p).tolist())

        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update with possibility {self.weight[sampled_client_indices]}."

        return message, sampled_client_indices

    def update_selected_clients(self, all_client=False):
        """Call "client_update" function of each selected client."""
        selected_total_size = 0

        if all_client:
            clients = self.clients
            message = f"All clients are updated (with total sample size: "
        else:
            clients = []
            for idx in self.sampled_clients_indices:
                clients.append(self.clients[idx])

            message = f"...{len(self.sampled_clients_indices)} clients are selected and updated (with total sample size: "

        for client in clients:
            client.client_update()
            selected_total_size += len(client)

        message += f"{str(selected_total_size)})!"
        return message

    def evaluate_selected_models(self):
        """Call "client_evaluate" function of each selected client."""
        for idx in self.sampled_clients_indices:
            self.clients[idx].client_evaluate()

        message = f"...finished evaluation of {str(self.sampled_clients_indices)} selected clients!"

        return message

    def transmit_model(self, model, to_all_clients=False):
        if to_all_clients:
            target_clients = self.clients
            message = f"...successfully transmitted models to all {str(self.num_clients)} clients!"
        else:
            target_clients = []
            for index in self.sampled_clients_indices:
                target_clients.append(self.clients[index])
            message = f"...successfully transmitted models to {str(len(self.sampled_clients_indices))} selected clients!"

        for target_client in target_clients:
            target_client.model = copy.deepcopy(model)
            target_client.global_model = copy.deepcopy(model)

        return message


