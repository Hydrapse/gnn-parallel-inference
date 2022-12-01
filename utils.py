import os.path as osp
import torch
from torch_sparse import fill_diag, mul
from torch_sparse import sum as sparsesum


def save_sub_data(data_list, save_dir, norm=False, mapping=False):
    norm_str = '_norm' if norm else ''
    mapping_str = '_mapping' if mapping else ''
    for i, sub_data in enumerate(data_list):
        filename = f'partition_{len(data_list)}_{i}{norm_str}{mapping_str}.pt'
        path = osp.join(save_dir, filename)
        torch.save(sub_data, path)


def load_sub_data(ptr_idx: int, num_partition: int, save_dir: str, norm=False, mapping=False):
    norm_str = '_norm' if norm else ''
    mapping_str = '_mapping' if mapping else ''
    filename = f'partition_{num_partition}_{ptr_idx}{norm_str}{mapping_str}.pt'
    path = osp.join(save_dir, filename)
    return torch.load(path)


def elapse(tik, tok):
    """input time.perf_count, output elapse time (ms)"""
    return f'{(tok - tik)*1000:.0f} ms'


def adj_norm(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t

