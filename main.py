from src.server import Server
from src.utils.Printer import Printer

from src.config import config
import numpy as np

def fedavg():
    fraction = [1]
    for f in fraction:
        accuracy = []
        for i in range(10):
            config.FRACTION = f
            config.RUN_NAME = config.RUN_NAME_ALL.format(f, i)

            # setup tensorboard and logging printer
            printer = Printer()
            printer.print("\n[WELCOME] ")
            tensorboard_writer = printer.get_tensorboard_writer()

            # initialize federated learning
            central_server = Server(tensorboard_writer)
            central_server.setup()

            # do federated learning
            accu = central_server.fit()
            accuracy.append(accu)
            printer.print(accu)

        accuracy = np.array(accuracy)
        printer.print(np.mean(accuracy,axis=0))

        # bye!
        printer.print("...done all learning process!\n...exit program!")


def our():
    fraction = [0.2]
    accuracy = []
    for f in fraction:
        for j in range(1):
            config.FRACTION = f
            config.RUN_NAME = config.RUN_NAME_ALL.format(f)

            # setup tensorboard and logging printer
            printer = Printer()
            printer.print("\n[WELCOME] ")
            tensorboard_writer = printer.get_tensorboard_writer()

            # initialize federated learning
            central_server = Server(tensorboard_writer)
            central_server.setup()

            # do federated learning
            accuracy.append(central_server.fit())

        printer.print(sum(accuracy) / len(accuracy))


if __name__ == "__main__":
    fedavg()