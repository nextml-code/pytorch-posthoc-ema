# %%
import posthoc_ema
import posthoc_ema.visualization


posthoc_ema.visualization.reconstruction_error(
    betas=(0.9, 0.99, 0.999),
    target_beta_range=(0.9, 0.95),
)
# %%
