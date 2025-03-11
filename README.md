# pytorch-posthoc-ema

Choose your EMA decay rate after training. No need to decide upfront.

The library uses `sigma_rel` (relative standard deviation) to parameterize EMA decay rates, which relates to the classical EMA decay rate `beta` as follows:

```python
beta = 0.3700  # Fast decay   -> sigma_rel ≈ 0.001
beta = 0.9055  # Medium decay -> sigma_rel ≈ 0.01
beta = 0.9680  # Medium decay -> sigma_rel ≈ 0.03
beta = 0.9808  # Medium decay -> sigma_rel ≈ 0.05
beta = 0.9911  # Slow decay   -> sigma_rel ≈ 0.10
beta = 0.9944  # Slow decay   -> sigma_rel ≈ 0.15
beta = 0.9962  # Slow decay   -> sigma_rel ≈ 0.20
beta = 0.9979  # Slow decay   -> sigma_rel ≈ 0.27
beta = 0.9999  # Very slow decay -> sigma_rel ≈ 0.40
```

This library was adapted from [ema-pytorch](https://github.com/lucidrains/ema-pytorch) by lucidrains.

New features and changes:

- No extra VRAM usage by keeping EMA on cpu
- No extra VRAM usage for EMA synthesis during evaluation
- Low RAM usage for EMA synthesis
- Simplified or more explicit usage
- Opinionated defaults
- Select number of checkpoints to keep
- Allow "Switch EMA" with PostHocEMA
- Visualization of EMA reconstruction error before training

## Install

```bash
pip install pytorch-posthoc-ema
```

or

```bash
poetry add pytorch-posthoc-ema
```

## Basic Usage

```python
import torch
from posthoc_ema import PostHocEMA

model = torch.nn.Linear(512, 512)

posthoc_ema = PostHocEMA.from_model(model, "posthoc-ema")

for _ in range(1000):
    # mutate your network, normally with an optimizer
    with torch.no_grad():
        model.weight.copy_(torch.randn_like(model.weight))
        model.bias.copy_(torch.randn_like(model.bias))

    posthoc_ema.update_(model)

data = torch.randn(1, 512)
predictions = model(data)

# use the helper
with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
    ema_predictions = ema_model(data)
```

### Load After Training

```python
# With model
posthoc_ema = PostHocEMA.from_path("posthoc-ema", model)
with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
    ema_predictions = ema_model(data)

# Without model
posthoc_ema = PostHocEMA.from_path("posthoc-ema")
with posthoc_ema.state_dict(sigma_rel=0.15) as state_dict:
    model.load_state_dict(state_dict, strict=False)
```

## Advanced Usage

### Switch EMA During Training

```python
with posthoc_ema.state_dict(sigma_rel=0.15) as state_dict:
    model.load_state_dict(state_dict, strict=False)
```

### Visualize Reconstruction Quality

```python
posthoc_ema.reconstruction_error()
```

### Configuration

```python
posthoc_ema = PostHocEMA.from_model(
    model,
    checkpoint_dir="path/to/checkpoints",
    max_checkpoints=20,  # Keep last 20 checkpoints per EMA model
    sigma_rels=(0.05, 0.28),  # Default relative standard deviations from paper
    update_every=10,  # Update EMA weights every 10 steps
    checkpoint_every=1000,  # Create checkpoints every 1000 steps
    checkpoint_dtype=torch.float16,  # Store checkpoints in half precision
)
```

## Citations

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696}
}
```

```bibtex
@article{Lee2024SlowAS,
    title   = {Slow and Steady Wins the Race: Maintaining Plasticity with Hare and Tortoise Networks},
    author  = {Hojoon Lee and Hyeonseo Cho and Hyunseung Kim and Donghu Kim and Dugki Min and Jaegul Choo and Clare Lyle},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.02596}
}
```

```bibtex
@article{Li2024SwitchEA,
    title   = {Switch EMA: A Free Lunch for Better Flatness and Sharpness},
    author  = {Siyuan Li and Zicheng Liu and Juanxi Tian and Ge Wang and Zedong Wang and Weiyang Jin and Di Wu and Cheng Tan and Tao Lin and Yang Liu and Baigui Sun and Stan Z. Li},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.09240}
}
```
