# %%
import posthoc_ema
import posthoc_ema.visualization


posthoc_ema.visualization.reconstruction_error(
    betas=(0.9, 0.95),
    target_beta_range=(0.8, 0.99),
)
# %%
