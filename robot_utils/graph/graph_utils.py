import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

def get_dense_edge_index(vertices):
    """
    given the number of nodes/vertices, return edge index with a dense connection
    Args:
        vertices: number of vertices

    Returns: edge index of a dense connection

    """
    idx = np.arange((vertices - 1) * vertices).reshape((vertices - 1, vertices))
    idx = idx + np.arange(1, vertices).reshape((-1, 1))
    idx = idx.flatten()

    row = idx // vertices
    col = idx % vertices
    edge_index = np.concatenate((row, col)).reshape(2, -1)
    # p = 0.1
    # mask = np.random.choice(a=[False, True], size=(edge_index.shape[1],), p=[p, 1 - p])
    return edge_index


def get_sparse_edge_index(vertices):
    """
    given the number of nodes/vertices, return edge index with a sparse connection, with a probability p = 0.2 for each
    edge to be ignored
    Args:
        vertices: number of nodes/vertices

    Returns: edge index of a sparse connection

    """
    l = []
    for i in range(vertices):
        for j in range(i + 1, vertices):
            l.append([i, j])
    idx = np.vstack(l).T
    p = 0.2
    while True:
        mask = np.random.choice(a=[False, True], size=(idx.shape[1],), p=[p, 1 - p])
        masked_idx = idx[:, mask]
        if len(set(masked_idx.flatten())) == vertices:
            break
    row, col = idx
    return np.concatenate((row, col, col, row)).reshape((2, -1))


def get_sequential_edge_index(vertices):
    """
    given the number of nodes/vertices, return edge index with a sequential connection
    Args:
        vertices: number of vertices

    Returns: edge index of a sequential connection

    """
    row = np.arange(vertices - 1)
    col = row + 1
    edge_index = np.concatenate((row, col, col, row)).reshape(2, -1)
    return edge_index


def get_tree_edge_index(vertices):
    """
    given the number of nodes/vertices, return edge index with a sequential connection
    Args:
        vertices: number of vertices

    Returns: edge index of a sequential connection

    """
    edge_index = get_sequential_edge_index(vertices)
    additional_index = []
    for i in range(2, vertices):
        for j in range(i - 1):
            additional_index.append([i, j])
    if not additional_index:
        return edge_index
    n_additional_edges = len(additional_index)
    edge_index = np.concatenate((edge_index, np.vstack(additional_index).T), axis=1)
    return edge_index, n_additional_edges


def scatter_add(x, index):
    sum_array = np.zeros((np.max(index)+1, x.shape[1]))
    for i in range(np.max(index)+1):
        sum_array[i, :] = np.sum(x[index == i], axis=0)
    return sum_array


def save_normalized_graph(dataset, saving_path, name="normalized_graph", mode="enyu"):
    """
    normalize all graphs in the 'dataset' and save the normalized graph to 'saving_path' with 'name_mode.pt'. The mode
    controls which features to be normalized.
    Args:
        dataset:
        saving_path:
        name:
        mode: default 'enyu'. 'e': edge attribute 'edge_attr', 'n': node 'x', 'y': target 'y', 'u': global feature 'u'

    Returns: the normalized graph

    """
    dataloader = DataLoader(dataset, batch_size=dataset.len())
    ng = Data(mode=mode)
    for i, g in enumerate(dataloader):
        if 'n' in mode:
            ng['x_mean'] = torch.mean(g['x'], dim=0)
            ng['x_std'] = torch.std(g['x'], dim=0)
        if 'e' in mode:
            ng['edge_attr_mean'] = torch.mean(g['edge_attr'], dim=0)
            ng['edge_attr_std'] = torch.std(g['edge_attr'], dim=0)
        if 'y' in mode:
            ng['y_mean'] = torch.mean(g['y'], dim=0)
            ng['y_std'] = torch.std(g['y'], dim=0)
        if 'u' in mode:
            ng['u_mean'] = torch.mean(g['u'], dim=0)
            ng['u_std'] = torch.std(g['u'], dim=0)

        torch.save(ng, os.path.join(saving_path, name+"_{}.pt".format(mode)))
    return ng


def normalize_graph(g, ng=None, mode=None):
    x = (g['x'] - ng['x_mean']) / (ng['x_std'] + 1e-6).to(device)
    edge_attr = (g['edge_attr'] - ng['edge_attr_mean']) / (ng['edge_attr_std'] + 1e-6).to(device)
    y = (g['y'] - ng['y_mean']) / (ng['y_std'] + 1e-6).to(device)
    # if ng and not mode:
    #     mode = ng['mode']
    # x = (g['x'] - ng['x_mean']) / (ng['x_std'] + 1e-6).to(device) if 'n' in mode \
    #     else g['x'].to(device)
    # edge_attr = (g['edge_attr'] - ng['edge_attr_mean']) / (ng['edge_attr_std'] + 1e-6).to(device) if 'e' in mode \
    #     else g['edge_attr'].to(device)
    # y = (g['y'] - ng['y_mean']) / (ng['y_std'] + 1e-6).to(device) if 'y' in mode \
    #     else g['y'].to(device)
    # u = (g['u'] - ng['u_mean']) / (ng['y_std'] + 1e-6).to(device) if 'u' in mode \
    #     else g['u'].to(device)
    return x, edge_attr, y


def prepare_graph_data(g, ng, device):
    if ng:
        x, edge_attr, y = normalize_graph(g, ng)
    else:
        x = g['x'].to(device)
        edge_attr = g['edge_attr'].to(device)
        y = g['y'].to(device)

    edge_index = g['edge_index'].to(device)
    u = g['u'].to(device)
    batch = g['batch'].to(device)
    return x, edge_index, edge_attr, u, batch, y

