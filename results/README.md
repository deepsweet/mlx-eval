# Evaluation of various MLX quantizations

Quantization aims to reduce the precision of a language model's parameters from higher to lower bit-widths. To measure the impact, we need to compare different metrics between a reference model and its quantized versions.

In our evaluation, the model works in an autoregressive fashion: it receives a sequence of tokens and, in a single forward pass, predicts a probability distribution over the entire vocabulary for the next token. You can think of this as a highly advanced autocomplete: the model considers every possible next token, assigns each a likelihood, and the token with the highest probability is its best guess of what comes next. By comparing these predicted probabilities to the actual next token in the text, we can measure the model's fidelity, uncertainty, and correctness.

## Metrics

### KLD

Kullback‑Leibler divergence (KLD) measures how much the whole probability distribution over the vocabulary for the next token has shifted.

The unit is *nats* (units of information), because the computation uses natural logarithms. Lower KLD indicates higher fidelity, perfect value is 0.

- `KLD` – mean per-token divergence
- `KLD p95` – 95th percentile – 95% of tokens have a divergence ≤ this value, and the worst 5% exceed it
- `KLD p99` – 99th percentile – 99% of tokens have a divergence ≤ this value, and the worst 1% exceed it

### PPL

Perplexity (PPL) measures the uncertainty in the correct next token, but *not necessarily the one it chooses*.

The value is unitless. Lower PPL indicates higher confidence, perfect value is 1. A PPL of 10 means the model is as uncertain, *on average*, as if it were randomly choosing among 10 equally likely options.

- `PPL` – mean perplexity
- `Δ PPL` – delta in mean perplexity, occasional negative values are an artifact of quantization noise acting as a "regularization filter" and do not indicate a genuine improvement

### Acc@1

Top‑1 accuracy (Acc@1) measures how often the most probable token matches the true token.

The value is unitless. Higher Acc@1 indicates higher hard‑decision accuracy, perfect value is 1. Acc@1 of 0.5 means the top prediction matches the true next token 50% of the time.

- `Acc@1` – mean accuracy
- `Δ Acc@1` – delta in mean accuracy, occasional positive values are typically within measurement noise and do not indicate a genuine improvement

### Further readings

- ["A Visual Guide to Quantization"](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- ["Why Maybe We're Measuring LLM Compression Wrong"](https://huggingface.co/blog/rishiraj/kld-guided-quantization)
- ["The 'Q4_K_M' Illusion: Why KL Divergence and Perplexity Are Your Only Friends in the GGUF Wild West"](https://www.banandre.com/blog/quantization-fidelity-benchmarking-kld-and-ppl-as-metrics-for-gguf-model-selection)

## Qwen3.6-35B-A3B

![Qwen3.6-35B-A3B KLD/RAM chart](./Qwen3.6-35B-A3B.svg)

### Reference

- model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- multimodal: Vision-Language
- data type: bfloat16
- prompt: 8 windows x 16384 tokens = 131072 tokens of Aes Sedai's multi-domain [combined_all_micro.txt](https://huggingface.co/AesSedai/GLM-4.5-GGUF/raw/main/combined_all_micro.txt)
- PPL: 5.578384
- Acc@1: 0.633111

### oQ

- ["oQ: oMLX Universal Dynamic Quantization"](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md)
- https://huggingface.co/collections/deepsweet/qwen36-35b-a3b
- tool: [oMLX](https://github.com/jundot/omlx) v0.3.6
- sensitivity model:
  - tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
  - quantization: Q8
  - mode: affine
  - group size: default, omitted
  - data type: bfloat16
- text-only: no
- data type: bfloat16

| Quant |      KLD |  KLD p95 |  KLD p99 |      PPL |     Δ PPL |    Acc@1 |   Δ Acc@1 |   RAM | 
|-------|---------:|---------:|---------:|---------:|----------:|---------:|----------:|------:|
| oQ2   | 0.293579 | 1.156250 | 2.812500 | 6.617926 | +1.039542 | 0.602126 | -0.030985 | 11.40 |
| oQ3   | 0.196899 | 0.726562 | 1.820312 | 6.361276 | +0.782892 | 0.611663 | -0.021448 | 14.77 |
| oQ3.5 | 0.191650 | 0.710938 | 1.789062 | 6.312603 | +0.734219 | 0.613601 | -0.019510 | 16.00 |
| oQ4   | 0.036423 | 0.142578 | 0.365234 | 5.668936 | +0.090552 | 0.631020 | -0.002091 | 18.83 |
| oQ5   | 0.013802 | 0.101074 | 0.155273 | 5.587953 | +0.009569 | 0.632760 | -0.000351 | 22.76 |
| oQ6   | 0.011642 | 0.095215 | 0.134766 | 5.577979 | -0.000404 | 0.632874 | -0.000237 | 26.51 |
| oQ8   | 0.007549 | 0.085938 | 0.125000 | 5.566999 | -0.011385 | 0.633553 | +0.000443 | 34.27 |

### Q

- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- mode: affine
- group size: default, omitted
- data type: bfloat16

| Quant |      KLD |  KLD p95 |   KLD p99 |       PPL |      Δ PPL |    Acc@1 |   Δ Acc@1 |   RAM | 
|-------|---------:|---------:|----------:|----------:|-----------:|---------:|----------:|------:|
| Q2    | 3.044922 | 7.562500 | 10.187500 | 99.345022 | +93.766638 | 0.292308 | -0.340803 | 10.10 |
| Q3    | 0.254028 | 0.902344 |  2.078125 |  6.615274 |  +1.036890 | 0.605727 | -0.027384 | 14.14 |
| Q4    | 0.069305 | 0.238281 |  0.605469 |  5.756796 |  +0.178412 | 0.627648 | -0.005463 | 18.17 |
| Q5    | 0.026352 | 0.120605 |  0.245117 |  5.575208 |  -0.003176 | 0.632859 | -0.000252 | 22.20 |
| Q6    | 0.012466 | 0.097656 |  0.137695 |  5.571558 |  -0.006826 | 0.632874 | -0.000237 | 26.23 |
| Q8    | 0.007668 | 0.085449 |  0.125000 |  5.582524 |  +0.004140 | 0.633393 | +0.000282 | 34.30 |

<sup>*Q2 is off the chart</sup>

### MXFP

- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- mode: mxfp4 / mxfp8
- group size: default, omitted
- data type: bfloat16

| Quant |      KLD |  KLD p95 |  KLD p99 |      PPL |     Δ PPL |    Acc@1 |   Δ Acc@1 |   RAM | 
|-------|---------:|---------:|---------:|---------:|----------:|---------:|----------:|------:|
| MXFP4 | 0.124512 | 0.404980 | 1.015625 | 5.894460 | +0.316076 | 0.621536 | -0.011574 | 17.16 |
| MXFP8 | 0.051361 | 0.180664 | 0.414062 | 5.712183 | +0.133799 | 0.628235 | -0.004875 | 33.29 |

### UD

- ["MLX Dynamic Quants"](https://unsloth.ai/docs/models/qwen3.6#mlx-dynamic-quants)
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit

| Quant |      KLD |  KLD p95 |  KLD p99 |      PPL |     Δ PPL |    Acc@1 |   Δ Acc@1 |   RAM | 
|-------|---------:|---------:|---------:|---------:|----------:|---------:|----------:|------:|
| UD3   | 0.058746 | 0.214844 | 0.617188 | 5.745224 | +0.166840 | 0.628319 | -0.004792 | 15.35 |
| UD4   | 0.020897 | 0.112793 | 0.213867 | 5.584422 | +0.006038 | 0.632561 | -0.000549 | 19.32 |
