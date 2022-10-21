import gc
import copy
from collections import OrderedDict

# custom packages
from .config import config
from .utils.DatasetController import *
logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        train: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, device, distribution):
        """client training configs"""
        self.batch_size = config.BATCH_SIZE
        self.local_epoch = config.LOCAL_EPOCH
        self.criterion = config.CRITERION
        self.optimizer = config.OPTIMIZER
        self.optim_config = config.OPTIMIZER_CONFIG

        """server side configs"""
        self.id = client_id
        self.device = device
        self.distribution = distribution
        self.temporal_heterogeneous = False
        self.train = None
        self.test = None
        self.__model = None
        self.global_model = None

        self.idx_t0 = 0.0
        self.idx_t1 = 0.0
        self.just_updated = False

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def get_gradient(self):
        grad = np.subtract(self.global_model.flatten_model(), self.model.flatten_model())

        return grad / (len(self.train) * self.local_epoch * config.OPTIMIZER_CONFIG['lr'] / self.batch_size)

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train)

    def mutate(self):
        self.temporal_heterogeneous = True

    def update_train(self, new_dataset, replace=False):
        if self.train is None or replace:
            self.train = new_dataset
        else:
            self.train + new_dataset

    def update_test(self, new_dataset, replace=True):
        if self.test is None or replace:
            self.test = new_dataset
        else:
            self.test + new_dataset

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        self.model.to("cpu")

        if self.just_updated:
            _, global_accuracy = self.client_evaluate(current_model=False, log=False)
            _, current_accuracy = self.client_evaluate(current_model=True, log=False)

            self.idx_t0 = current_accuracy - global_accuracy
            self.just_updated = False

    def client_evaluate(self, current_model=True, log=True, test_set=True):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        if current_model:
            model = self.model
        else:
            model = self.global_model

        if test_set:
            dataset = self.test
        else:
            dataset = self.train

        model.eval()
        model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataset.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        model.to("cpu")

        test_loss = test_loss / len(dataset.get_dataloader())
        test_accuracy = correct / len(dataset)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\
            \n\t=> Distribution: {self.distribution}\n"
        if log:
            print(message, flush=True); logging.info(message)

        del message; gc.collect()

        return test_loss, test_accuracy

    def get_performance_gap(self):
        _, global_accuracy = self.client_evaluate(current_model=False, log=False)
        _, current_accuracy = self.client_evaluate(current_model=True, log=False)

        self.idx_t1 = current_accuracy - global_accuracy

        # print(id(self.model))
        # print(id(self.global_model))

        if self.idx_t0 is None:
            self.idx_t0 = self.idx_t1
        else:
            res = self.idx_t1 - self.idx_t0

            message = f"\t[Client {str(self.id).zfill(4)}]:!\
                \n\t=> Global Accuracy: {100. * global_accuracy:.2f}%\
                \n\t=> Current Accuracy: {100. * current_accuracy:.2f}%\
                \n\t=> idx_t0: {100. * self.idx_t0: .2f}%\
                \n\t=> idx_t1: {100. * self.idx_t1: .2f}%"

            print(message, flush=True);
            logging.info(message)

            return max(res, 0.000001)



