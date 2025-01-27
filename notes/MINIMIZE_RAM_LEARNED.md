# Learnings from RAM Optimization in PostHocEMA

## What We Tried

### Effective Strategies

1. Processing parameters one at a time instead of all at once
2. Moving operations to CPU to avoid VRAM spikes
3. Aggressive memory cleanup with `torch.cuda.empty_cache()`
4. Avoiding `deepcopy` where possible
5. Using state dictionaries instead of full model copies

### Less Effective Strategies

1. Processing checkpoints sequentially - didn't help much since we still need all weights for synthesis
2. Checkpoint pruning - the synthesis algorithm needs all checkpoints for accurate results
3. Batch processing parameters - added complexity without significant memory savings
4. Using reduced precision (float16) - memory savings were minimal compared to algorithmic improvements

## Current Bottlenecks

1. State Dictionary Management

   - Need to keep full state dict in memory during synthesis
   - Each parameter requires memory for both original and synthesized values

2. Weight Calculation
   - Requires loading all checkpoints to solve the linear system
   - Matrix operations for weight calculation can be memory intensive

## Future Optimization Ideas

1. Streaming Parameter Updates

   - Load and process one parameter at a time from checkpoints
   - Challenge: Need to maintain consistency across parameters

2. Partial Model Updates

   - Allow updating only specific layers/parameters
   - Could reduce memory when only part of model needs EMA

3. In-place Operations

   - More aggressive use of in-place operations for parameter updates
   - Challenge: Need to ensure numerical stability

4. Checkpoint Compression
   - Store checkpoints in compressed format
   - Challenge: Decompression time vs memory tradeoff

## Key Insights

1. The synthesis algorithm fundamentally requires all checkpoints to produce accurate results
2. Memory usage scales with both model size and number of checkpoints
3. CPU operations are slower but help avoid VRAM spikes
4. The biggest memory spikes occur during:
   - Initial model copying
   - State dictionary creation
   - Weight synthesis

## Recommendations

1. Keep synthesis operations on CPU when possible
2. Use state dictionaries instead of full model copies
3. Process one parameter at a time
4. Clean up memory aggressively
5. Consider the tradeoff between synthesis accuracy and memory usage when choosing number of checkpoints
