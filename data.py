import os
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import mask_to_index
from torch_geometric_autoscale import metis, permute, SubgraphLoader
from torch_geometric.datasets import Reddit2

from utils import load_sub_data, adj_norm, save_sub_data

NUM_PARTITION = 4
PRE_NORM = True
PRE_MAPPING = True


if __name__ == '__main__':
    # path = os.path.join(os.environ.get('DATA_DIR'), 'ogb')
    # dataset = PygNodePropPredDataset(name='ogbn-products', root=path, transform=T.ToSparseTensor())
    # data = dataset[0]

    path = os.path.join(os.environ.get('DATA_DIR'), 'pyg', 'Reddit2')
    dataset = Reddit2(path, transform=T.ToSparseTensor())
    data = dataset[0]

    if PRE_NORM:
        data.adj_t = adj_norm(data.adj_t)

    perm, ptr = metis(data.adj_t, NUM_PARTITION, log=True)
    data = permute(data, perm, log=True)
    data_list = list(SubgraphLoader(data, ptr, batch_size=1, shuffle=False))

    if PRE_MAPPING:
        halo_nodes = [[[] for j in range(NUM_PARTITION)] for i in range(NUM_PARTITION)]
        reverse_idx = [[] for _ in range(NUM_PARTITION)]
        halo_counts = [[] for _ in range(NUM_PARTITION)]

        package_sizes = []
        for batch_id, sub_data in enumerate(data_list):
            out_of_batch = sub_data.n_id[sub_data.batch_size:]

            package_size = 0
            for i in range(NUM_PARTITION):
                if batch_id == i:
                    continue
                mask = (out_of_batch >= ptr[i]) & (out_of_batch < ptr[i + 1])
                num_halos = mask.sum().item()
                if num_halos > package_size:
                    package_size = num_halos

                halo_nodes[i][batch_id] = out_of_batch[mask] - ptr[i]
                reverse_idx[batch_id].append(mask_to_index(mask))
                halo_counts[batch_id].append(num_halos)

            package_sizes.append(package_size)

        for i, sub_data in enumerate(data_list):
            data_list[i] = (sub_data, halo_nodes[i], torch.cat(reverse_idx[i]), halo_counts[i], package_sizes)

    save_sub_data(data_list, dataset.processed_dir, PRE_NORM, PRE_MAPPING)

    sub_data = load_sub_data(0, NUM_PARTITION, dataset.processed_dir, PRE_NORM, PRE_MAPPING)
    print(sub_data)
