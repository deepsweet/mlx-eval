import gc
import pathlib
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def run_reference(ref_model_path, prompt_text, max_tokens):
    """
    Run a reference model on the prompt and return log-probabilities,
    tokenized prompt, and reference perplexity as an MLX array.
    """

    mlx.core.clear_cache()

    print("Loading model...")
    ref_model, ref_tokenizer = mlx_lm.load(ref_model_path)

    # tokenize prompt and truncate to max_tokens with no padding
    token_ids = ref_tokenizer.encode(prompt_text, truncation=True, max_length=max_tokens)
    token_len = len(token_ids)

    if token_len < max_tokens:
        raise ValueError(f"Prompt {token_len} < max_tokens {max_tokens}")

    # add batch dimension (batch_size, max_tokens)
    prompt = mlx.core.array(token_ids)[None]

    print("Calculating log-probabilities...")
    # raw logits per token from forward pass over vocabulary (batch_size, max_tokens, vocab_size)
    logits = ref_model(prompt)

    del ref_model
    gc.collect()
    mlx.core.clear_cache()

    # convert logits to numerically stable log-probabilities along the vocabulary axis
    log_probs = mlx.nn.log_softmax(logits, axis=-1)

    print("Calculating perplexity...")
    # drop last token because there is no "next token" to predict
    shift_logits = logits[:, :-1, :]
    # drop first token because there is no previous token to use as context for prediction
    shift_prompt = prompt[:, 1:]
    # cross-entropy loss between the predicted logits and target tokens
    cross_entropy = mlx.nn.losses.cross_entropy(shift_logits, shift_prompt, reduction="mean")
    # convert cross-entropy to perplexity
    ppl_mean = mlx.core.exp(cross_entropy).item()

    print("Calculating top-1 accuracy...")
    # top-1 accuracy: fraction of tokens where the predicted token matches the true next token
    top1_preds = mlx.core.argmax(shift_logits, axis=-1)
    top1_acc = mlx.core.mean(top1_preds == shift_prompt).item()

    return {
        "prompt": prompt,
        "log_probs": log_probs,
        "ppl_mean": ppl_mean,
        "top1_acc": top1_acc,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: mlx_eval.reference <reference_model_path> <max_tokens>")
        sys.exit(1)

    ref_model_path = sys.argv[1]
    max_tokens = int(sys.argv[2])

    print("Loading prompt...")
    prompt_text = pathlib.Path(const.PROMPT_PATH).read_text(encoding="utf-8")

    result = run_reference(ref_model_path, prompt_text, max_tokens)

    print("Saving outputs...")
    mlx.core.savez(
        const.OUTPUTS_PATH,
        prompt=result["prompt"],
        log_probs=result["log_probs"],
        ppl_mean=mlx.core.array(result["ppl_mean"]),
        top1_acc=mlx.core.array(result["top1_acc"]),
    )

    print(f"\nPPL mean: {result["ppl_mean"]:.6f}")
    print(f"Top-1 accuracy: {result["top1_acc"]:.6f}")


if __name__ == "__main__":
    main()
