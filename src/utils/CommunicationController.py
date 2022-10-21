import numpy as np
import logging
import copy
import gc

# custom packages
from ..config import config
from ..utils.Printer import *


class CommunicationController:
    def __init__(self, clients):
        self.weight = None
        self.improvement = None
        self.num_clients = len(clients)
        self.clients = clients

        self.sampled_clients_indices = None

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


