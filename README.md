# pytorch post-hoc ema

> This is a work in progress.

The PyTorch Post-hoc EMA library improves neural network performance by applying Exponential Moving Average (EMA) techniques after training. This approach allows for the adjustment of EMA profiles post-training, which is crucial for optimizing model weight stabilization without predefining decay parameters.

By implementing the post-hoc synthesized EMA method from Karras et al., the library offers flexibility in exploring EMA profiles' effects on training and sampling. It seamlessly integrates with PyTorch models, making it easy to enhance machine learning projects with post-hoc EMA adjustments.

This library was adapted from [ema-pytorch](https://github.com/lucidrains/ema-pytorch) by lucidrains.

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

ema = PostHocEMA(model)

for _ in range(1000):

    # mutate your network, normally with an optimizer
    with torch.no_grad():
        model.weight.copy_(torch.randn_like(model.weight))
        model.bias.copy_(torch.randn_like(model.bias))

    ema.update()

data = torch.randn(1, 512)
output = model(data)

with ema.model(sigma_rel=0.15) as ema_model:
    ema_output = ema_model(data)
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
