import gc
import statistics
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def run_compare(
    target_model_path,
    prompt,
    ref_log_probs,
    ref_ppl_mean,
    ref_top1_acc,
):
    """
    Run a target model comparison using reference log-probabilities and tokenized prompt,
    and return KL divergence metrics, perplexity, and memory usage.
    """

    mlx.core.clear_cache()

    print("Loading model...")
    memory_before = mlx.core.get_active_memory()
    target_model = mlx_lm.load(target_model_path)[0]
    memory_after = mlx.core.get_active_memory()
    target_memory = memory_after - memory_before

    print("Calculating log-probabilities...")
    # raw logits per token from forward pass over vocabulary (batch_size, max_tokens, vocab_size)
    logits = target_model(prompt)

    del target_model
    gc.collect()
    mlx.core.clear_cache()

    # convert logits to numerically stable log-probabilities along the vocabulary axis
    log_probs = mlx.nn.log_softmax(logits, axis=-1)

    print("Calculating KL divergence...")
    # per-token KL Divergence summed over vocabulary (batch_size, max_tokens)
    kld_none = mlx.nn.losses.kl_div_loss(log_probs, ref_log_probs, reduction="none")
    kld_mean = mlx.core.mean(kld_none).item()
    kld_list = kld_none.flatten().tolist()
    kld_p95 = statistics.quantiles(kld_list, n=100)[-5]
    kld_p99 = statistics.quantiles(kld_list, n=100)[-1]

    print("Calculating perplexity...")
    # drop last token because there is no "next token" to predict
    shift_logits = logits[:, :-1, :]
    # drop first token because there is no previous token to use as context for prediction
    shift_prompt = prompt[:, 1:]
    # cross-entropy loss between the predicted logits and target tokens
    cross_entropy = mlx.nn.losses.cross_entropy(shift_logits, shift_prompt, reduction="mean")
    # convert cross-entropy to perplexity
    ppl_mean = mlx.core.exp(cross_entropy).item()
    ppl_delta = ppl_mean - ref_ppl_mean

    print("Calculating top-1 accuracy...")
    # top-1 accuracy: fraction of tokens where the predicted token matches the true next token
    top1_preds = mlx.core.argmax(shift_logits, axis=-1)
    top1_acc = mlx.core.mean(top1_preds == shift_prompt).item()
    top1_delta = top1_acc - ref_top1_acc

    return {
        "kld_mean": kld_mean,
        "kld_p95": kld_p95,
        "kld_p99": kld_p99,
        "ppl_mean": ppl_mean,
        "ppl_delta": ppl_delta,
        "top1_acc": top1_acc,
        "top1_delta": top1_delta,
        "memory": target_memory,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: mlx_eval.compare <target_model_path>")
        sys.exit(1)

    target_model_path = sys.argv[1]

    print("Loading reference outputs...")
    data = mlx.core.load(const.OUTPUTS_PATH)

    prompt = data["prompt"]
    ref_log_probs = data["log_probs"]
    ref_ppl_mean = data["ppl_mean"].item()
    ref_top1_acc = data["top1_acc"].item()

    result = run_compare(
        target_model_path,
        prompt,
        ref_log_probs,
        ref_ppl_mean,
        ref_top1_acc,
    )

    print(f"\nKLD mean: {result["kld_mean"]:.6f}")
    print(f"KLD p95: {result["kld_p95"]:.6f}")
    print(f"KLD p99: {result["kld_p99"]:.6f}")
    print(f"PPL mean: {result["ppl_mean"]:.6f}")
    print(f"PPL delta: {result["ppl_delta"]:+.6f}")
    print(f"Top-1 accuracy: {result["top1_acc"]:.6f}")
    print(f"Top-1 delta: {result["top1_delta"]:+.6f}")

    target_model_memory_gib = result["memory"] / (1024**3)
    print(f"RAM: {target_model_memory_gib:.2f}")


if __name__ == "__main__":
    main()
