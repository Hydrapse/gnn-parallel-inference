# gnn-parallel-inference

### Goal
Perform full-batch GCN inference with multiple GPUs

### Workflow
- Pre-Processing: `data.py`
  1. Normalize the Adjacency Matrix
  2. Partition the original graph
  3. Compute mappings for halo vertices (which part of halo vertices belongs to which cluster)
- Training
- Inference: `main.py`
  1. Apply Linear Transform to core vertices
  2. Use `torch.distributed.gather` to collect features of halo vertices
  3. Use SpMM to aggregate neighbor information for core vertices
