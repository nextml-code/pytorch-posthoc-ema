# pytorch post-hoc ema

> This is a work in progress.

The PyTorch Post-hoc EMA library improves neural network performance by applying Exponential Moving Average (EMA) techniques after training. This approach allows for the adjustment of EMA profiles post-training, which is crucial for optimizing model weight stabilization without predefining decay parameters.

By implementing the post-hoc synthesized EMA method from Karras et al., the library offers flexibility in exploring EMA profiles' effects on training and sampling. It seamlessly integrates with PyTorch models, making it easy to enhance machine learning projects with post-hoc EMA adjustments.

This library was adapted from [ema-pytorch](https://github.com/lucidrains/ema-pytorch) by lucidrains.

Why?

- Simplified or more explicit usage
- Opinionated defaults

New features:

- Select number of checkpoints to keep
- Switch EMA also with PostHocEMA
- Low VRAM usage by keeping EMA on cpu
- Low VRAM synthesization

TODO:

- [ ] Investigate best options for saving checkpoints
- [ ] Implement new usage
- [ ] Add tests
- [ ] Optimize vram usage

## Install

```bash
poetry add pytorch-posthoc-ema
```

## Usage

```python
import torch
from posthoc_ema import PostHocEMA

model = torch.nn.Linear(512, 512)

posthoc_ema = PostHocEMA(
    "posthoc-ema",
    model,
)

for _ in range(1000):

    # mutate your network, normally with an optimizer
    with torch.no_grad():
        model.weight.copy_(torch.randn_like(model.weight))
        model.bias.copy_(torch.randn_like(model.bias))

    posthoc_ema.update(model)

data = torch.randn(1, 512)
predictions = model(data)

# use the helper
with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
    ema_predictions = ema_model(data)

# or without magic
model.cpu()

ema_state_dict = posthoc_ema.state_dict(sigma_rel=0.15) as state_dict:
ema_model = deepcopy(model)
ema_model.load_state_dict(state_dict)
ema_predictions = ema_model(data)
del ema_model
```

Synthesize after training:

```python
posthoc_ema = PostHocEMA(
    "posthoc-ema",
    model,
)

with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
    ema_predictions = ema_model(data)
```

Or without model:

```python
posthoc_ema = PostHocEMA("posthoc-ema")

ema_state_dict = posthoc_ema.state_dict(sigma_rel=0.15)
```

## Citations

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696},
    url     = {https://api.semanticscholar.org/CorpusID:265659032}
}
```

```bibtex
@article{Lee2024SlowAS,
    title   = {Slow and Steady Wins the Race: Maintaining Plasticity with Hare and Tortoise Networks},
    author  = {Hojoon Lee and Hyeonseo Cho and Hyunseung Kim and Donghu Kim and Dugki Min and Jaegul Choo and Clare Lyle},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.02596},
    url     = {https://api.semanticscholar.org/CorpusID:270258586}
}
```

```bibtex
@article{Li2024SwitchEA,
    title   = {Switch EMA: A Free Lunch for Better Flatness and Sharpness},
    author  = {Siyuan Li and Zicheng Liu and Juanxi Tian and Ge Wang and Zedong Wang and Weiyang Jin and Di Wu and Cheng Tan and Tao Lin and Yang Liu and Baigui Sun and Stan Z. Li},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.09240},
    url     = {https://api.semanticscholar.org/CorpusID:267657558}
}
```
