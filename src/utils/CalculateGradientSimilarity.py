import os
import torch
import numpy as np
import pandas


def flatten_model(model_dict):
    tensor = np.array([])

    for key in model_dict.keys():
        tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))

    return torch.tensor(tensor).squeeze()


def get_cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_gradients(name):
    path = os.path.join('../../models', name)
    if not os.path.exists(path):
        raise Exception('Unable to find model: {}'.format(name))

    num_files = len(os.listdir(path))
    previous = None

    gradients = []
    for i in range(1, num_files + 1):
        model_path = os.path.join(path, name +'_{}.pth'.format(i))
        model_dict = torch.load(model_path)['model']
        current = flatten_model(model_dict)
        if previous is not None:
            gradients.append(current - previous)
        previous = current

    return gradients


def calculate_gradient_similarity(name_1, name_2):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    g1 = get_gradients(name_1)
    g2 = get_gradients(name_2)

    cossims = []
    for i in range(min(len(g1), len(g2))):
        cossim = get_cossim(g1[i], g2[i])
        cossims.append(cossim)

    df = pandas.DataFrame(cossims)
    df.to_csv('../../models/{}_{}.csv'.format(name_1, name_2), index=False)


calculate_gradient_similarity('new', 'fedavg')

