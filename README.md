# mlx-eval

Utilities to evaluate KL divergence and perplexity of MLX quantizations.

See [detailed results](./results) for more information:

![Qwen3.6-35B-A3B KLD/RAM chart](./results/Qwen3.6-35B-A3B.svg)

## Usage

```sh
# clone the repo
git clone git@github.com:deepsweet/mlx-eval.git
cd mlx-eval/

# install dependencies
uv sync

# download an original reference model
uv tool install huggingface-hub
hf download Qwen/Qwen3.6-35B-A3B

# convert it into MLX, add `--dtype float16` if needed
# either text-only:
uv tool install mlx-lm
mlx_lm.convert \
  --hf-path ~/.cache/huggingface/models/Qwen/Qwen3.6-35B-A3B \
  --mlx-path /path/to/Qwen3.6-35B-A3B-MLX
# or multimodal:
uv tool install mlx-vlm --with torchvision
mlx_vlm.convert \
  --hf-path ~/.cache/huggingface/models/Qwen/Qwen3.6-35B-A3B \
  --mlx-path /path/to/Qwen3.6-35B-A3B-MLX

# prepare a diverse prompt, Aes Sedai's "combined_all_micro" would suffice
curl -L "https://huggingface.co/AesSedai/GLM-4.5-GGUF/raw/main/combined_all_micro.txt" > prompt.txt

# load, calculate and save reference model log-probabilities
# mlx_eval.reference <reference_model_path> <window_count> <max_tokens>
uv run mlx_eval.reference /path/to/Qwen3.6-35B-A3B-MLX 8 16384

# compare a target quantized model against it
# mlx_eval.compare <target_model_path> <window_count>
uv run mlx_eval.compare /path/to/Qwen3.6-35B-A3B-MLX-oQ8 8

# cleanup when finished
rm outputs-*.npz
```

## Generate chart

```sh
uv run results/<model_name>.py
```

## Lint and test

```sh
uv sync --group dev
uv run ruff check .
uv run pytest .
```
