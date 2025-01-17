# %%
import posthoc_ema
import posthoc_ema.visualization

posthoc_ema.visualization.reconstruction_error(
    sigma_rels=(0.15, 0.28),
    target_sigma_rel_range=(0.1, 0.3),
    max_checkpoints=50,
)
# %%
