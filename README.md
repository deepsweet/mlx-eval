# mlx-eval

Utilities to evaluate MLX quantizations.

See [detailed results](./results) for more information:

![Qwen3.6-35B-A3B KLD/RAM chart](./results/Qwen3.6-35B-A3B.svg)

## Usage

```sh
# clone the repo
git clone git@github.com:deepsweet/mlx-eval.git
cd mlx-eval/

# install dependencies
uv sync

# prepare an original reference MLX model 
# fof example, text-only using mlx-lm, or multimodal using mlx-vlm:
uv tool install mlx-vlm --with torchvision
mlx_vlm.convert \
  --hf-path Qwen/Qwen3.6-35B-A3B \
  --mlx-path /path/to/Qwen3.6-35B-A3B-MLX

# prepare a quantized target MLX model
# for example:
mlx_vlm.convert \
  --hf-path Qwen/Qwen3.6-35B-A3B \
  --mlx-path /path/to/Qwen3.6-35B-A3B-MLX-Q4 \
  --quantize \
  --q-bits 4

# compute and store the reference model data into outputs/
# mlx_eval.reference <reference_model_path> <window_count> <max_tokens>
uv run mlx_eval.reference /path/to/Qwen3.6-35B-A3B-MLX 16 8192

# and compare the target quantized model against it
# mlx_eval.compare <target_model_path> <window_count>
uv run mlx_eval.compare /path/to/Qwen3.6-35B-A3B-MLX-Q4 16
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

## License

MIT.

The evaluation prompt is derived from [Aes Sedai](https://huggingface.co/AesSedai)'s
`combined_all_micro.txt`.
