from src.server import Server
from src.utils.Printer import Printer

from src.config import config
import numpy as np
import pandas


def main():
    res = {}
    for f in config.fractions:
        accuracy = []
        uploads = []
        for i in range(config.run_time):
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
            accu, round_uploads = central_server.fit()
            accuracy.append(accu)
            uploads.append(round_uploads)
            printer.print(accu)

        accuracy = np.array(accuracy)
        accuracy = np.mean(accuracy,axis=0)
        uploads = np.array(uploads)
        uploads = np.mean(uploads,axis=0)
        res['f={}'.format(np.mean(uploads))] = accuracy
        res['uploads_f={}'.format(np.mean(uploads))] = uploads
        printer.print(accuracy)
        printer.print(uploads)

    if config.SAVE:
        df = pandas.DataFrame(res)
        df.to_csv('{}_{}_class={}_test.csv'.format(config.DATASET_NAME, config.RUN_TYPE, config.NUM_SELECTED_CLASS), index=False)
    # bye!
    printer.print("...done all learning process!\n...exit program!")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    main()