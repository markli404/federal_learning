import configparser
import copy
import logging
import random
import timeit

import numpy as np
import pandas
import torch
from torch.nn import CosineSimilarity
import scipy
from tqdm.auto import tqdm
from collections import OrderedDict

# custom packages
from .models import *
from .client import Client
from .utils.DriftController import *
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
        self.runtype = config.RUN_TYPE

        self.model = eval(config.MODEL_NAME)(**config.MODEL_CONFIG)

        self.seed = config.SEED
        self.device = config.DEVICE
        self.cos = CosineSimilarity(dim=0, eps=1e-6)

        self.data = None
        self.dataloader = None
        self.round_upload = []
        self.round_accuracy = []

        # drift related
        self.drift_clients = []
        self.source_classes = []
        self.target_classes = []

        # scaffold
        self.c_global = None

        # FGT
        self.sop_cache = None

    def log(self, message):
        message = f"[Round: {str(self._round).zfill(4)}] " + message
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    def setup(self):
        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model)

        self.log(
            f"...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")

        # initialize DatasetController
        self.DatasetController = DatasetController()
        self.log('...sucessfully initialized dataset controller for [{}]'.format(config.DATASET_NAME))

        # initialize DriftController
        self.DriftController = DriftController(self.DatasetController)
        self.log('...sucessfully initialized drift controller.')

        # create clients
        self.clients = self.create_clients()

        # initialize CommunicationController
        self.CommunicationController = CommunicationController(self.clients)

        # send the model skeleton to all clients
        message = self.CommunicationController.transmit_model(self.model, to_all_clients=True)

        self.log(message)

    def create_clients(self, distribution=None):
        clients = []
        distribution = np.zeros((self.num_clients, config.NUM_CLASS))
        # np.random.seed(config.NUM_SELECTED_CLASS)
        client_idx = np.arange(self.num_clients)
        np.random.shuffle(client_idx)
        for i in client_idx:
            class_pool = np.sum(distribution, axis=0)
            class_pool = np.where(class_pool < int(config.NUM_SELECTED_CLASS * self.num_clients / config.NUM_CLASS))[0]
            try:
                selected_class = np.random.choice(class_pool, config.NUM_SELECTED_CLASS, replace=False)
            except:
                selected_class = np.random.choice(np.arange(config.NUM_CLASS),
                                                  config.NUM_SELECTED_CLASS - len(class_pool), replace=False)
                selected_class = np.append(selected_class, class_pool)

            #TODO
            if i == 0:
                selected_class = np.arange(10)

            for j in selected_class:
                distribution[i][j] = 1

                if i == 0:
                    distribution[i][j] = 1 / 10 * config.NUM_SELECTED_CLASS

        distribution = distribution / config.NUM_SELECTED_CLASS
        for i in range(self.num_clients):
            d = distribution[i]
            client = Client(client_id=i, device=self.device, distribution=distribution[i])
            clients.append(client)

        self.log(f"...successfully created all {str(self.num_clients)} clients!")
        return clients

    def aggregate_models(self, sampled_client_indices, coefficients):
        self.log(f"...with the weights of {str(coefficients)}.")
        new_model = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].client_current.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

        self.model.to("cpu")
        for key in self.model.state_dict().keys():
            averaged_weights[key] += self.model.state_dict()[key] * (1 - np.sum(coefficients))
        self.model.to(self.device)

        new_model.load_state_dict(averaged_weights)
        return new_model

    def aggregate_models_with_cache(self, sampled_client_indices, coeff):
        new_model = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()

        models = []

        for i in range(self.num_clients):
            if i in sampled_client_indices:
                models.append(self.clients[i].client_current)
            else:
                if self.clients[i].client_previous is not None:
                    models.append(self.clients[i].client_previous)

        weight = 1 / len(models)

        for i, model in enumerate(models):
            local_weights = model.state_dict()
            for key in self.model.state_dict().keys():
                if i == 0:
                    averaged_weights[key] = weight * local_weights[key]
                else:
                    averaged_weights[key] += weight * local_weights[key]

        new_model.load_state_dict(averaged_weights)
        return new_model

    def aggregate_models_scaffold(self, sampled_client_indices, coeff):
        total_delta = copy.deepcopy(self.model.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            c_delta_para = self.clients[idx].c_delta_para
            for key in total_delta:
                total_delta[key] += c_delta_para[key]

        for key in total_delta:
            total_delta[key] = total_delta[key] / len(sampled_client_indices)

        for i in sampled_client_indices:
            client = self.clients[i]
            c_global_para = client.c_global.state_dict()
            for key in c_global_para:
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    # print(c_global_para[key].type())
                    c_global_para[key] += total_delta[key]

            client.c_global.load_state_dict(c_global_para)
        return self.aggregate_models(sampled_client_indices, coeff)

    def aggregate_models_with_pruning(self, sampled_client_indices, coeff, eps=0.001):
        def get_cossim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        def gradient_calibration(gradient, sum_of_gradient):
            projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
            if projection > 0:
                gradient = gradient + projection * sum_of_gradient
            else:
                gradient = gradient - 2 * projection * sum_of_gradient
            return gradient

        def layerwise_calibration(gradient, sum_of_gradient):
            gradient = self.model.unflatten_model(gradient)
            sum_of_gradient = self.model.unflatten_model(sum_of_gradient)

            for key in gradient.keys():
                g_layer = gradient[key].numpy().flatten()
                sog_layer = sum_of_gradient[key].numpy().flatten()
                shape = gradient[key].shape
                # cos_sim = get_cossim(g_layer, sog_layer)
                # print(key, cos_sim)
                if cos_sim < 0.3:
                    projection = np.dot(g_layer, sog_layer) / np.linalg.norm(sog_layer) ** 2
                    if projection > 0:
                        g_layer = g_layer + projection * sog_layer
                    else:
                        g_layer = g_layer - 2 * projection * sog_layer

                    # cos_sim = get_cossim(g_layer, sog_layer)
                    # print(key, cos_sim)

                g_layer = torch.tensor(g_layer.reshape(shape))
                gradient[key] = g_layer

            return self.model.flatten_model(gradient)


        def aggregate_model(sum_of_gradient):
            new_model = copy.copy(self.model)
            new_model.to('cpu')
            new_weights = new_model.state_dict()
            global_gradient = self.model.unflatten_model(sum_of_gradient)
            for key in new_model.state_dict().keys():
                new_weights[key] = new_weights[key] - 1 * global_gradient[key]

            new_model.load_state_dict(new_weights)
            return new_model

        # get gradient (delta) of selected clients
        gradients = []
        iid = self.clients[0].get_gradient()
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        # get sum of gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient
        if self.sop_cache is None:
            self.sop_cache = sum_of_gradient - 2 * 0.001

        # test the change of calibration
        print('sum of gradient before calibration', get_cossim(iid, sum_of_gradient))

        while np.linalg.norm(sop - self.sop_cache) > eps and True:
            for i, g in enumerate(gradients):

                cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
                if cos_sim < 0.8:
                    # iid_cossim = get_cossim(g, iid)
                    # print(i, iid_cossim)
                    calibrated_gradient = layerwise_calibration(g, sop)
                    gradients[i] = calibrated_gradient
                    cos_sim_new = np.dot(sop, calibrated_gradient) / (np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                    # iid_cossim = get_cossim(calibrated_gradient, iid)
                    # print(i, iid_cossim)

            self.sop_cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            sop = sum_of_gradient



        # client = self.clients[0]
        # client_current = copy.deepcopy(client.client_current)
        # client_current = client_current.state_dict()
        # client_global = copy.deepcopy(client.global_current)
        #
        # averaged_weights = client.get_gradient()
        #
        # new_model = client_global.unflatten_model(averaged_weights)
        # client_global.to('cpu')
        # client_global = client_global.state_dict()
        # for key in client_global.keys():
        #     client_global[key] = client_global[key] - new_model[key]
        #     a = np.subtract(client_global[key], client_current[key])
        #     print(a)
        print('sum of gradient before calibration', get_cossim(iid, sum_of_gradient))
        new_model = aggregate_model(sum_of_gradient)

        return new_model

    def calculate_similarity(self, model_1, model_2):
        # tensor_1 = model_1.flatten_model()
        tensor_1 = model_1
        tensor_2 = model_2.flatten_model()
        assert (tensor_1.shape[0] == tensor_2.shape[0])

        return self.cos(tensor_1, tensor_2).numpy().item()

    def get_perfect_coeff_LS(self, sampled_client_indices):
        D = []
        for i in sampled_client_indices:
            client = self.clients[i]
            D.append(client.distribution)

        penalty = 0.3

        A = np.array(D).T
        I = np.identity(len(sampled_client_indices)) * penalty
        A = np.vstack([A, I])

        b = np.ones(config.NUM_CLASS) / config.NUM_CLASS * len(sampled_client_indices)
        p = np.ones(len(sampled_client_indices)) * penalty
        b = np.hstack([b, p])

        coeff = scipy.optimize.nnls(A, b)[0]
        coeff = coeff / np.sum(coeff) * len(sampled_client_indices) / self.num_clients

        d = np.matmul(np.array(D).T, coeff)
        print(d)
        return coeff

    def get_perfect_coeff_MC(self, sampled_client_indices):
        A = []
        for i in sampled_client_indices:
            client = self.clients[i]
            A.append(client.distribution)

        A = np.array(A).T
        target_coeff = 1 / self.num_clients
        target_distribution = np.ones(config.NUM_CLASS) / config.NUM_CLASS
        best_coeff = None
        best_residual = 100000
        best_distribution = None
        for i in range(1000000):
            coeff = [random.uniform(target_coeff * 0.4, target_coeff * 1.6) for j in range(len(sampled_client_indices))]
            current_distribution = np.matmul(A, coeff)
            current_distribution = current_distribution / np.sum(current_distribution)
            residual = np.linalg.norm(current_distribution - target_distribution)
            if residual < best_residual:
                best_residual = residual
                best_coeff = coeff
                best_distribution = current_distribution
        best_coeff = best_coeff / np.sum(best_coeff) * len(sampled_client_indices) / self.num_clients
        return best_coeff

    def get_blanced_coeff(self, sampled_client_indices):
        num_of_drifts = 0
        for i in sampled_client_indices:
            if self.clients[i].drift:
                num_of_drifts += 1

        coeff = []
        for i in sampled_client_indices:
            if self.clients[i].drift:
                coeff.append(1 / num_of_drifts)
            else:
                coeff.append(1 / (len(sampled_client_indices) - num_of_drifts))

        coeff = coeff / np.sum(coeff) * len(sampled_client_indices) / self.num_clients
        return coeff

    def get_uniformed_coeff(self, sampled_client_indices):
        return np.ones(len(sampled_client_indices)) / len(sampled_client_indices)

    def update_model(self, sampled_client_indices, coeff_method, update_method):
        """Average the updated and transmitted parameters from each selected client."""
        if not sampled_client_indices:
            message = f"None of the clients were selected"
            self.round_upload.append(0)
            self.log(message)
            return

        message = f"Updating {sampled_client_indices} clients...!"
        self.log(message)
        self.round_upload.append(len(sampled_client_indices))

        coeff = coeff_method(sampled_client_indices)
        self.log(message)
        self.model = update_method(sampled_client_indices, coeff)


    def train_without_drift(self, sample_method, coeff_method, update_method, update_type=config.RUN_TYPE):
        # assign new training and test set based on distribution
        self.DriftController.enforce_drift(self.clients, None)
        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(update_type, all_client=True)
        self.log(message)

        message, sampled_client_indices = sample_method()

        self.log(message)

        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_all_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices, coeff_method, update_method)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def train_with_drift_after_converge(self, sample_method, coeff_method, update_method, update_type=config.RUN_TYPE, drift_type=config.DRIFT_TYPE):
        # assign new training and test set based on distribution
        self.DriftController.enforce_drift(self.clients, drift_type)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(update_type, all_client=True)
        self.log(message)

        message, sampled_client_indices = sample_method()

        for client in self.clients:
            if client.id in sampled_client_indices:
                client.global_previous = client.global_current
                client.client_previous = client.client_current
                client.test_previous = client.test
                client.freshness = 1
            else:
                client.freshness -= 0.4

        self.log(message)

        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_all_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices, coeff_method, update_method)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)


    def train_fedavg(self):
        """Do federated training."""
        # assign new training and test set based on distribution
        self.enforce_drift()
        self.DatasetController.update_clients_datasets(self.clients, self.drift_clients, self.source_classes,
                                                       self.target_classes, True)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=True)
        self.log(message)

        # select clients based on weights
        if self._round < 15:
            message, sampled_client_indices = self.CommunicationController.sample_all_clients()
        else:
            message, sampled_client_indices = self.CommunicationController.sample_clients_random()
        self.log(message)

        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_selected_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def save_model(self):
        path = os.path.join('models', self.runtype)
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, self.runtype + '_' + str(self._round) + '.pth')
        torch.save({'model': self.model.state_dict()}, path)


    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        # calculate the sample distribution of all clients
        global_distribution, global_test_set = self.get_test_dataset()

        message = pretty_list(global_distribution)
        self.log(f"Current test set distribution: [{str(message)}]. ")
        # start evaluation process
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        correct_per_class = np.zeros(config.NUM_CLASS)
        total_per_class = np.zeros(config.NUM_CLASS)
        with torch.no_grad():
            for data, labels in global_test_set.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(config.CRITERION)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                labels = labels.cpu().numpy()
                predicted = predicted.cpu().numpy().flatten()
                for i in range(config.NUM_CLASS):
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
            if client.drift or everyone:
                if addition:
                    client.distribution += distribution
                else:
                    client.distribution = distribution

                self.log(f"Client {str(client.id)} has a shifted distribution: {str(client.distribution)}")

    def get_test_dataset(self):
        global_distribution = np.zeros(config.NUM_CLASS)
        for client in self.clients:
            global_distribution += client.distribution
        global_distribution = global_distribution / sum(global_distribution)

        global_test_set = None
        for client in self.clients:
            if global_test_set is None:
                global_test_set = copy.deepcopy(client.test)
            else:
                global_test_set + client.test

        return global_distribution, global_test_set


    def class_swap(self, client, class_1=1, class_2=2):
        client.train.class_swap(class_1, class_2)
        client.test.class_swap(class_1, class_2)


    def fit(self):
        """Execute the whole process of the federated learning."""
        for r in range(config.NUM_ROUNDS):
            self._round = r + 1

            # train the model
            """
            train method takes following parameters
                sample_method: method used to select client in the update
                coeff_method: method used to calculate coeff for model aggregation
                    get_uniformed_coeff(sampled_client_indices)
            """
            if False: #self._round < config.DRIFT[0]:
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_all_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models, update_type='fedavg', drift_type='None')
            elif config.RUN_TYPE == 'fedavg':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE in ['case_study', 'case_study_test']:
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients_casestudy,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'fedpns':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients_fed_pns,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'fast':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients_TSF,
                    self.get_perfect_coeff_LS,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'fedprox':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'scaffold':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models_scaffold)
            elif config.RUN_TYPE == 'new':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_models_with_pruning)
            else:
                raise Exception("No federal learning method is found.")

            # save model to evaluate gradient similarity
            if config.SAVE_MODEL:
                self.save_model()

            # evaluate the model
            self.evaluate_global_model()

            message = f"Clients have uploaded their model {str(sum(self.round_upload))} times！"
            self.log(message)

            message = f"Overall Accuracy is {str(sum(self.round_accuracy) / len(self.round_accuracy))}!"
            self.log(message)

        self.writer.add_text('accuracy', str(sum(self.round_accuracy) / len(self.round_accuracy)))
        self.writer.add_text('freq', str(sum(self.round_upload)))

        return self.round_accuracy, self.round_upload
