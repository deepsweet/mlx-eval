import gc
import math
import pathlib
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def run_reference(ref_model_path, token_ids):
    """
    Run a reference model on the prompt and return log-probabilities,
    tokenized prompt, and reference perplexity as an MLX array.
    """

    mlx.core.clear_cache()

    print("Loading model...")
    model = mlx_lm.utils.load_model(ref_model_path)[0]

    # add batch dimension (batch_size, max_tokens)
    prompt = mlx.core.array(token_ids)[None]

    print("Calculating log-probabilities...")
    # raw logits per token from forward pass over vocabulary (batch_size, max_tokens, vocab_size)
    logits = model(prompt)

    # materialise logits, break giant lazy graph
    mlx.core.eval(logits)

    print("Unloading model...")
    # cleanup
    del model
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
    if len(sys.argv) != 4:
        print("Usage: mlx_eval.reference <reference_model_path> <window_count> <max_tokens>")
        sys.exit(1)

    ref_model_path = pathlib.Path(sys.argv[1])
    window_count = int(sys.argv[2])
    max_tokens = int(sys.argv[3])

    print("Loading prompt...")
    prompt_text = pathlib.Path(const.PROMPT_PATH).read_text(encoding="utf-8")
    tokenizer = mlx_lm.utils.load_tokenizer(ref_model_path)
    total_tokens = max_tokens * window_count

    prompt = tokenizer.encode(
        prompt_text,
        truncation=True,
        max_length=total_tokens,
        add_special_tokens=False,
    )

    token_len = len(prompt)

    if token_len < total_tokens:
        raise ValueError(f"Prompt {token_len} < max_tokens {max_tokens}")

    ref_ppls = []
    ref_top1s = []

    for i in range(window_count):
        start = i * max_tokens
        end = start + max_tokens
        token_ids = prompt[start:end]

        print(f"\nProcessing window {i + 1}/{window_count}")
        result = run_reference(ref_model_path, token_ids)

        mlx.core.savez(
            f"outputs-{i:02d}.npz",
            prompt=result["prompt"],
            log_probs=result["log_probs"],
            ppl_mean=mlx.core.array(result["ppl_mean"]),
            top1_acc=mlx.core.array(result["top1_acc"]),
        )

        ref_ppls.append(result["ppl_mean"])
        ref_top1s.append(result["top1_acc"])

        print(f"PPL mean: {result['ppl_mean']:.6f}")
        print(f"Top-1 accuracy: {result['top1_acc']:.6f}")

        del result
        gc.collect()
        mlx.core.clear_cache()

    del tokenizer
    gc.collect()
    mlx.core.clear_cache()

    # token-weighted: all windows have identical prediction counts
    overall_ppl = math.exp(sum(math.log(p) for p in ref_ppls) / len(ref_ppls))
    overall_top1 = sum(ref_top1s) / len(ref_top1s)

    mlx.core.savez(
        "outputs-overall.npz",
        ppl_mean=mlx.core.array(overall_ppl),
        top1_acc=mlx.core.array(overall_top1),
    )

    print(f"\nPPL mean: {overall_ppl:.6f}")
    print(f"Top-1 accuracy: {overall_top1:.6f}")


if __name__ == "__main__":
    main()
