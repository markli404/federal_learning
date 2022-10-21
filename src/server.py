import copy
import logging
import timeit

import numpy as np
import torch
from torch.nn import CosineSimilarity

from tqdm.auto import tqdm
from collections import OrderedDict

# custom packages
from .models import *
from .client import Client
from .utils.CommunicationController import *
from .utils.DatasetController import *
from .utils.Printer import *
logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self, writer):
        self._round = 0
        self.clients = None
        self.writer = writer
        self.num_clients = config.NUM_CLIENTS
        self.runtype = config.RUNTYPE

        self.model = eval(config.MODEL_NAME)(**config.MODEL_CONFIG)

        self.seed = config.SEED
        self.device = config.DEVICE
        self.cos = CosineSimilarity(dim=0, eps=1e-6)

        self.data = None
        self.dataloader = None
        self.num_upload = 0
        self.round_accuracy = []

    def log(self, message):
        message = f"[Round: {str(self._round).zfill(4)}] " + message
        print(message); logging.info(message)
        del message; gc.collect()

    def setup(self):
        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model)

        self.log(f"...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")

        # initialize DatasetController
        self.DatasetController = DatasetController()
        self.log('...sucessfully initialized dataset controller for [{}]'.format(config.DATASET_NAME))

        # create clients
        if config.CLASS_SWAP:
            initial_distribution = np.ones(10) / 10
        else:
            initial_distribution = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0])

        initial_distribution = None
        self.clients = self.create_clients(initial_distribution)
        # select mutate clients
        # self.select_drifted_clients(4)

        # initialize CommunicationController
        self.CommunicationController = CommunicationController(self.clients)
        
        # send the model skeleton to all clients
        message = self.CommunicationController.transmit_model(self.model, to_all_clients=True)
        self.log(message)

    def create_clients(self, distribution):
        clients = []
        for k in range(self.num_clients):
            if distribution is None:
                num = 4
                initial_distribution = np.zeros(10)
                idxs = np.random.choice(10, num, replace=False)
                for idx in idxs:
                    initial_distribution[idxs] = 1 / num

                client = Client(client_id=k, device=self.device, distribution=initial_distribution)
                x = (initial_distribution[1] != 0)
                client.temporal_heterogeneous = x
            else:
                client = Client(client_id=k, device=self.device, distribution=distribution)
            clients.append(client)

        self.log(f"...successfully created all {str(self.num_clients)} clients!")
        return clients


    def select_drifted_clients(self, n):
        indices = np.random.choice(self.num_clients, n, replace=False)

        for index in indices:
            self.clients[index].mutate()

        self.log(f"Clients {str(indices)} will drift!")

    def aggregate_models(self, sampled_client_indices, coefficients, with_previous_model=True):
        new_model = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

        if with_previous_model:
            self.model.to("cpu")
            for key in self.model.state_dict().keys():
                # averaged_weights[key] += self.model.state_dict()[key] * (1 - config.MODEL_COEFF * len(sampled_client_indices))
                averaged_weights[key] += self.model.state_dict()[key] * (1 - config.MODEL_COEFF)
            self.model.to(self.device)

        new_model.load_state_dict(averaged_weights)
        return new_model

    def calculate_similarity(self, model_1, model_2):
        # tensor_1 = model_1.flatten_model()
        tensor_1 = model_1
        tensor_2 = model_2.flatten_model()
        assert(tensor_1.shape[0] == tensor_2.shape[0])

        return self.cos(tensor_1, tensor_2).numpy().item()

    def update_model(self, sampled_client_indices):
        """Average the updated and transmitted parameters from each selected client."""
        if not sampled_client_indices:
            message = f"None of the clients were selected"
            self.log(message)
            return

        message = f"Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        self.log(message)
        self.num_upload += len(sampled_client_indices)

        fedavg_coeff = [len(self.clients[idx]) for idx in sampled_client_indices]
        fedavg_coeff = np.array(fedavg_coeff) / sum(fedavg_coeff)
        fedavg_model = self.aggregate_models(sampled_client_indices, fedavg_coeff, with_previous_model=False)

        if self.runtype in ['fedavg', 'case_study']:
            self.model = fedavg_model
            message = f"...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        else:
            current_round_similarities = []
            last_round_similarities = []

            for idx in sampled_client_indices:
                current_round_similarities.append(self.calculate_similarity(self.clients[idx].get_gradient(), fedavg_model))
                last_round_similarities.append(self.calculate_similarity(self.clients[idx].get_gradient(), self.model))

            similarities = np.array(current_round_similarities) + np.array(last_round_similarities)
            # similarities = similarities / sum(similarities) * (config.MODEL_COEFF * len(sampled_client_indices))
            similarities = similarities / sum(similarities) * config.MODEL_COEFF
            self.model = self.aggregate_models(sampled_client_indices, similarities)

            message = f"...updated weights of {len(sampled_client_indices)} clients are successfully updated based on coeff: {pretty_list(similarities)}!"

        self.log(message)

    def train_federated_model_test(self):
        """Do federated training."""
        # assign new training and test set based on distribution
        # (self._round >= config.DRIFT * config.NUM_ROUNDS)
        self.DatasetController.update_clients_datasets(self.clients, False)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=True)
        self.log(message)

        # update client selection weight
        message = self.CommunicationController.update_weight()
        self.log(message)

        # select clients based on weights
        if self._round <= 2:
            message, sampled_client_indices = self.CommunicationController.sample_clients()
        else:
            message, sampled_client_indices = self.CommunicationController.sample_clients_test()
        self.log(message)

        for index in sampled_client_indices:
            self.clients[index].just_updated = True

        # evaluate selected clients with local dataset
        message = self.CommunicationController.evaluate_selected_models()
        self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model, to_all_clients=True)
        self.log(message)

    def train_case_study(self):
        """Do federated training."""
        # assign new training and test set based on distribution
        # (self._round >= config.DRIFT * config.NUM_ROUNDS)
        self.DatasetController.update_clients_datasets(self.clients, (self._round >= config.DRIFT * config.NUM_ROUNDS))

        # select clients based on weights
        message, sampled_client_indices = self.CommunicationController.sample_clients_casestudy()
        self.log(message)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=False)
        self.log(message)

        # evaluate selected clients with local dataset
        message = self.CommunicationController.evaluate_selected_models()
        self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices)

    def train_federated_model(self):
        """Do federated training."""
        # assign new training and test set based on distribution
        self.DatasetController.update_clients_datasets(self.clients, (self._round >= config.DRIFT * config.NUM_ROUNDS))

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=True)
        self.log(message)

        # update client selection weight
        message = self.CommunicationController.update_weight()
        self.log(message)

        # select clients based on weights
        if self._round <= 2:
            message, sampled_client_indices = self.CommunicationController.sample_clients()
        else:
            message, sampled_client_indices = self.CommunicationController.sample_clients_test()
        self.log(message)

        # evaluate selected clients with local dataset
        message = self.CommunicationController.evaluate_selected_models()
        self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def train_fedavg(self):
        """Do federated training."""
        # assign new training and test set based on distribution
        # (self._round >= config.DRIFT * config.NUM_ROUNDS)
        self.DatasetController.update_clients_datasets(self.clients, (self._round >= config.DRIFT * config.NUM_ROUNDS))

        # select clients based on weights
        message, sampled_client_indices = self.CommunicationController.sample_clients_random()
        self.log(message)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=False)
        self.log(message)

        # evaluate selected clients with local dataset
        message = self.CommunicationController.evaluate_selected_models()
        self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices)

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        # calculate the sample distribution of all clients
        global_distribution = np.zeros(config.NUM_CLASS)
        for client in self.clients:
            global_distribution += client.distribution
        global_distribution = global_distribution / sum(global_distribution)


        global_test_set = None
        for client in self.clients:
            if global_test_set is None:
                global_test_set = client.test
            else:
                global_test_set + client.test

        message = pretty_list(global_distribution)
        self.log(f"Current test set distribution: [{str(message)}]. ")

        # start evaluation process
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        correct_per_class = np.zeros(10)
        total_per_class = np.zeros(10)
        with torch.no_grad():
            for data, labels in global_test_set.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(config.CRITERION)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                labels = labels.cpu().numpy()
                predicted = predicted.cpu().numpy().flatten()
                for i in range(10):
                    c = np.where(labels == i)[0].tolist()
                    if not c:
                        continue
                    total_per_class[i] += len(c)
                    predicted_i = predicted[c]
                    predicted_correct = np.where(predicted_i == i)[0]
                    correct_per_class[i] += len(predicted_correct)
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        class_accuracy = []
        for i in range(len(total_per_class)):
            try:
                class_accuracy.append(correct_per_class[i] / total_per_class[i])
            except:
                class_accuracy.append(0)
        class_accuracy = ["%.2f" % i for i in class_accuracy]

        # calculate the metrics
        test_loss = test_loss / len(global_test_set.get_dataloader())
        test_accuracy = correct / len(global_test_set)
        self.round_accuracy.append(test_accuracy)

        # print to tensorboard and log
        self.writer.add_scalar('Loss', test_loss, self._round)
        self.writer.add_scalar('Accuracy', test_accuracy, self._round)

        message = f"Evaluate global model's performance...!\
            \n\t[Server] ...finished evaluation!\
            \n\t=> Loss: {test_loss:.4f}\
            \n\t=> Accuracy: {100. * test_accuracy:.2f}%\
            \n\t=> Class Accuracy: {class_accuracy}\n"
        self.log(message)

    def update_client_distribution(self, distribution, addition=False, everyone=False):
        for client in self.clients:
            if client.temporal_heterogeneous or everyone:
                if addition:
                    client.distribution += distribution
                else:
                    client.distribution = distribution

                self.log(f"Client {str(client.id)} has a shifted distribution: {str(client.distribution)}")

    def class_swap(self, client, class_1=1, class_2=2):
        client.train.class_swap(class_1, class_2)
        client.test.class_swap(class_1, class_2)

    def fit(self):
        """Execute the whole process of the federated learning."""
        for r in range(config.NUM_ROUNDS):
            self._round = r + 1

            # assign new distribution to drfited clients
            if config.DRIFT * config.NUM_ROUNDS == self._round:
                config.FRACTION = 0.3
                if config.CLASS_SWAP:
                    drift = []
                    for client in self.clients:
                        self.class_swap(client)
                        if client.temporal_heterogeneous:
                            drift.append(client.id)

                    print(drift)
                else:
                    new_dist = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
                    new_dist = np.array(new_dist) / sum(new_dist)
                    self.update_client_distribution(new_dist, addition=False, everyone=False)

                # weight = []
                # for client in self.clients:
                #     if client.temporal_heterogeneous:
                #         weight.append(5)
                #     else:
                #         weight.append(1)
                #
                # weight = np.array(weight) / sum(weight)
                # self.CommunicationController.weight = weight

            if int(0.25 * config.NUM_ROUNDS) == self._round and config.RUNTYPE == 'our':
                self.CommunicationController.transmit_model(model=self.model, to_all_clients=True)

            # train the model
            if config.RUNTYPE == 'fedavg':
                self.train_fedavg()
            elif config.RUNTYPE == 'case_study':
                self.train_case_study()
            elif config.RUNTYPE == 'our':
                self.train_federated_model_test()

            # evaluate the model
            self.evaluate_global_model()

            message = f"Clients have uploaded their model {str(self.num_upload)} times！"
            self.log(message)

            message = f"Overall Accuracy is {str(sum(self.round_accuracy) / len(self.round_accuracy))}!"
            self.log(message)

        self.writer.add_text('accuracy', str(sum(self.round_accuracy) / len(self.round_accuracy)))
        self.writer.add_text('freq', str(self.num_upload))

        return self.round_accuracy
