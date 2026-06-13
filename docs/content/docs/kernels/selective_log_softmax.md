---
title: Selective Log Softmax
description: Gathered log-softmax values without materializing full log-probabilities.
---

Selective log softmax computes one log-probability per row:

$$\text{out}[i] = \text{logits}[i, \text{index}[i]] - \log \sum_j \exp(\text{logits}[i, j])$$

This is equivalent to:

```python
logprobs = logits.log_softmax(dim=-1)
torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
```

The kernel is useful for workloads that only need the log-probability of selected vocabulary entries, such as scoring target tokens, reward-model traces, or preference/evaluation pipelines. It avoids materializing the full `[..., vocab_size]` log-probability tensor.

## Kernel design

### Online log-sum-exp

The kernel flattens every leading dimension into rows and streams across the final vocabulary dimension. Each row computes a numerically stable log-sum-exp with fp32 online accumulation:

1. Each thread scans a strided slice of the vocabulary and maintains a local maximum and sum.
2. Warp reductions combine local partials into warp-level max/sum pairs.
3. A final warp reduces those partials into one row-level log-sum-exp.
4. The selected logit is loaded once and written as `selected - logsumexp`.

This keeps the computation to one output value per row while avoiding a full `log_softmax` allocation.

### Shape and dtype constraints

`logits` must be contiguous with shape `[..., vocab_size]`, and `index` must be contiguous with shape `[...]`. The kernel supports `float16`, `bfloat16`, and `float32` logits with `int32` or `int64` indices.

### Comparison with `torch.compile`

`torch.compile` can optimize the expression, but the operation still naturally wants to form or reason about the full log-softmax surface. Greyhound's kernel is specialized for the selected-value case: it streams the vocabulary, computes only the log-sum-exp needed for normalization, and writes one scalar per row.

### Backward pass

`selective_log_softmax` is currently forward-only. The functional API raises `NotImplementedError` for backward.

## Usage

### Functional API

```python
import torch
from greyhound.nn.functional import selective_log_softmax

logits = torch.randn(8, 2048, 128256, device="cuda", dtype=torch.bfloat16)
index = torch.randint(0, logits.shape[-1], logits.shape[:-1], device="cuda")

selected_logprobs = selective_log_softmax(logits, index)
```

`greyhound.nn.functional.selective_log_softmax`

## Benchmarks

Benchmarks use bfloat16 inputs and a batch size of 8. The x-axis is sequence length and the y-axis is vocabulary size.

<div class="gpu-benchmark-picker" data-gpu-picker>
  <label class="gpu-picker-control">
    <span>GPU</span>
    <select data-gpu-select aria-label="Benchmark GPU"></select>
  </label>

  <div data-gpu-panel data-gpu="NVIDIA GeForce RTX 4090">

<iframe class="plot-frame" title="Selective log softmax forward speed" src="../../assets/plots_html/logprobs_forward_speed_batch_size=8_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Selective log softmax forward memory usage" src="../../assets/plots_html/logprobs_forward_memory_batch_size=8_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA H100 80GB HBM3" data-gpu-hidden>

<iframe class="plot-frame" title="Selective log softmax forward speed on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/logprobs_forward_speed_batch_size=8_dtype=bfloat16_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Selective log softmax forward memory usage on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/logprobs_forward_memory_batch_size=8_dtype=bfloat16_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA B200" data-gpu-hidden>

<iframe class="plot-frame" title="Selective log softmax forward speed on NVIDIA B200" src="../../assets/plots_html/logprobs_forward_speed_batch_size=8_dtype=bfloat16_NVIDIA_B200.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Selective log softmax forward memory usage on NVIDIA B200" src="../../assets/plots_html/logprobs_forward_memory_batch_size=8_dtype=bfloat16_NVIDIA_B200.html" scrolling="no"></iframe>

  </div>
</div>

<script type="module" src="../../assets/gpu-picker.js"></script>
