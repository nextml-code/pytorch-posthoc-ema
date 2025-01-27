# Minimizing RAM Usage in PostHocEMA

## What Worked

1. **Processing Parameters One at a Time**

   - Instead of loading all parameters from all checkpoints at once, process one parameter at a time
   - Free memory after processing each parameter with `del` and `torch.cuda.empty_cache()`
   - This significantly reduced peak memory usage by avoiding having multiple copies of large tensors in memory

2. **Avoiding `deepcopy`**

   - Removed `deepcopy` usage in `KarrasEMA.__init__` by creating a new model instance and loading state dict
   - Removed `deepcopy` in `PostHocEMA.model` context manager by storing original state and restoring it after use
   - This helped avoid having multiple copies of the model in memory

3. **Moving Operations to CPU**
   - Performing synthesis calculations on CPU to avoid VRAM spikes
   - Using `cpu()` when loading checkpoint tensors
   - Using `cpu()` for weight calculations

## What Didn't Help Much

1. **Processing Checkpoints Sequentially**

   - Initially tried loading and processing checkpoints one at a time
   - Didn't significantly reduce memory usage because the bottleneck was in parameter handling, not checkpoint loading

2. **Checkpoint Pruning**
   - Tried keeping only a limited number of checkpoints
   - Memory spikes were more related to how we processed the checkpoints rather than how many we kept

## Current Bottlenecks

1. **State Dictionary Management**

   - Still need to store at least one full copy of the model state during synthesis
   - Required for maintaining model functionality during context manager usage

2. **Weight Calculation**
   - Need to load all checkpoints to calculate optimal weights
   - Can't process weights incrementally due to the nature of the least squares solution

## Future Optimization Ideas

1. **Streaming Parameter Updates**

   - Could potentially stream parameters directly from checkpoints without loading full state dicts
   - Would require modifying checkpoint storage format

2. **Partial Model Updates**

   - Could implement partial model updates for cases where only specific layers need EMA
   - Would reduce memory usage but require changes to the API

3. **In-place Operations**
   - Could investigate more opportunities for in-place tensor operations
   - Need to be careful about maintaining correctness with in-place updates
