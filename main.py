import os
import time
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel

from torch_sparse import matmul, fill_diag
from torch_sparse import sum as sparsesum
from torch_geometric.nn import Linear, GCNConv

from torch_geometric.datasets import Reddit2
from ogb.nodeproppred import PygNodePropPredDataset

from utils import elapse, load_sub_data, adj_norm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 rank, world_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.world_size = world_size

        # TODO: CUDA Sync and Async
        #   https://pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations
        # self.s = torch.cuda.Stream()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, weight_initializer='glorot'))
        for _ in range(self.num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels, weight_initializer='glorot'))
        self.lins.append(Linear(hidden_channels, out_channels, weight_initializer='glorot'))

    @torch.no_grad()
    def forward(self, x, adj_t, halo_nodes, rev_idx, halo_counts, pkg_sizes, is_normed=False):
        if not is_normed:
            adj_t = adj_norm(adj_t)
        for i, lin in enumerate(self.lins):
            x = lin(x)

            # gather remote data
            if i > 0:  # assume data has been collected in the first layer
                def empty_pkg():
                    return torch.empty(pkg_sizes[self.rank], x.size(-1), dtype=x.dtype, device=self.rank)
                futs, rets = [], [empty_pkg() for _ in range(self.world_size)]
                for j in range(self.world_size):
                    if self.rank == j:
                        fut = dist.gather(empty_pkg(), gather_list=rets, dst=j, async_op=True)
                    else:
                        pkg = x[halo_nodes[j]]
                        fut = dist.gather(F.pad(pkg, (0, 0, 0, pkg_sizes[j]-pkg.size(0))),
                                          dst=j, async_op=True)
                    futs.append(fut)
                for fut in futs:
                    fut.wait()

                x_halo = torch.empty(adj_t.size(1) - adj_t.size(0), x.size(-1), dtype=x.dtype, device=self.rank)
                rets.pop(self.rank)
                x_halo[rev_idx] = torch.cat([ret[:hc] for hc, ret in zip(halo_counts, rets)], dim=0)
                x = torch.cat([x, x_halo], dim=0)

            x = matmul(adj_t, x)  # cu-Sparse

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)


def run(rank, world_size, processed_dir, num_classes):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.manual_seed(12345)

    if rank == 0:
        start = time.perf_counter()

    sub_data, halo_nodes, reverse_idx, halo_counts, package_sizes = \
        load_sub_data(rank, world_size, processed_dir, norm=True, mapping=True)
    data = sub_data.data
    dist.barrier()

    if rank == 0:
        end = time.perf_counter()
        print(f'Data Loading Time (Disk->Memory):', elapse(start, end))
        start = time.perf_counter()

    x, adj_t, y = data.x.to(rank), data.adj_t.to(rank), data.y.to(rank)
    model = GCN(data.x.size(-1), 128, num_classes, rank, world_size, num_layers=4).to(rank)
    dist.barrier()

    if rank == 0:
        end = time.perf_counter()
        print(f'Data Transfer Time (CPU->GPU):',  elapse(start, end))

    # model = DistributedDataParallel(model, device_ids=[rank])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        model.eval()

        if rank == 0:
            start = time.perf_counter()

        # optimizer.zero_grad()
        out = model(x, adj_t, halo_nodes, reverse_idx, halo_counts, package_sizes, is_normed=True)
        # loss = F.nll_loss(out, y.squeeze(1))
        # loss.backward()
        # optimizer.step()

        dist.barrier()

        if rank == 0:
            end = time.perf_counter()
            print(f'Epoch: {epoch:03d}, Time:{elapse(start, end)}')

    dist.destroy_process_group()


if __name__ == '__main__':
    path = os.environ.get('DATA_DIR')

    """pyg datasets"""
    path = os.path.join(path, 'pyg', 'Reddit2')
    dataset = Reddit2(path)

    """ogb datasets"""
    # path = os.path.join(path, 'ogb')
    # dataset = PygNodePropPredDataset(name='ogbn-papers100M', root=path)
    # dataset = PygNodePropPredDataset(name='ogbn-products', root=path)

    num_gpu = torch.cuda.device_count()
    print('Let\'s use', num_gpu, 'GPUs!')
    mp.spawn(run, args=(num_gpu, dataset.processed_dir, dataset.num_classes), nprocs=num_gpu, join=True)
