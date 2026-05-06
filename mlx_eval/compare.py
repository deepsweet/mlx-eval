import gc
import math
import pathlib
import statistics
import sys

import mlx.core
import mlx.nn
import mlx_lm


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
    model = mlx_lm.utils.load_model(target_model_path)[0]
    memory_after = mlx.core.get_active_memory()
    memory = memory_after - memory_before

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
        "kld_list": kld_list,
        "kld_p95": kld_p95,
        "kld_p99": kld_p99,
        "ppl_mean": ppl_mean,
        "ppl_delta": ppl_delta,
        "top1_acc": top1_acc,
        "top1_delta": top1_delta,
        "memory": memory,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: mlx_eval.compare <target_model_path> <window_count>")
        sys.exit(1)

    target_model_path = pathlib.Path(sys.argv[1])
    window_count = int(sys.argv[2])

    total_pred = 0
    log_ppl_weighted = 0
    correct_weighted = 0
    kld_weighted = 0
    memory_gib = 0
    all_kld = []

    for i in range(window_count):
        if i > 0:
            print()

        print(f"Processing window {i + 1}/{window_count}")

        ref_data = mlx.core.load(f"outputs-{i:02d}.npz")
        prompt = ref_data["prompt"]
        ref_log_probs = ref_data["log_probs"]
        ref_ppl_mean = ref_data["ppl_mean"].item()
        ref_top1_acc = ref_data["top1_acc"].item()

        result = run_compare(
            target_model_path,
            prompt,
            ref_log_probs,
            ref_ppl_mean,
            ref_top1_acc,
        )

        # number of next-token predictions in this window (sequence length - 1)
        num_pred = prompt.shape[1] - 1
        # accumulate total number of predictions across all windows
        total_pred += num_pred
        # accumulate token-weighted log-perplexity
        log_ppl_weighted += math.log(result["ppl_mean"]) * num_pred
        # accumulate token-weighted top-1 accuracy
        correct_weighted += result["top1_acc"] * num_pred
        # accumulate token-weighted mean KLD
        kld_weighted += result["kld_mean"] * num_pred
        # collect every per-token KLD value from this window for exact overall percentiles
        all_kld.extend(result["kld_list"])

        print(f"KLD mean: {result['kld_mean']:.6f}")
        print(f"PPL mean: {result['ppl_mean']:.6f}")
        print(f"PPL delta: {result['ppl_delta']:.6f}")
        print(f"Top-1 accuracy: {result['top1_acc']:.6f}")
        print(f"Top-1 accuracy delta: {result['top1_delta']:.6f}")

        if i == 0:
            memory_gib = result["memory"] / (1024**3)

        del ref_data, prompt, ref_log_probs, result
        gc.collect()
        mlx.core.clear_cache()

    # overall token-weighted average metrics
    overall_ppl = math.exp(log_ppl_weighted / total_pred)
    overall_top1 = correct_weighted / total_pred
    overall_kld_mean = kld_weighted / total_pred

    # pool all per-token KLD values for exact overall percentiles
    overall_kld_p95 = statistics.quantiles(all_kld, n=100)[-5]
    overall_kld_p99 = statistics.quantiles(all_kld, n=100)[-1]

    overall_ref = mlx.core.load("outputs-overall.npz")
    ref_ppl_overall = overall_ref["ppl_mean"].item()
    ref_top1_overall = overall_ref["top1_acc"].item()

    ppl_delta = overall_ppl - ref_ppl_overall
    top1_delta = overall_top1 - ref_top1_overall

    print(f"\nKLD mean: {overall_kld_mean:.6f}")
    print(f"KLD p95: {overall_kld_p95:.6f}")
    print(f"KLD p99: {overall_kld_p99:.6f}")
    print(f"PPL mean: {overall_ppl:.6f}")
    print(f"PPL delta: {ppl_delta:+.6f}")
    print(f"Top-1 accuracy: {overall_top1:.6f}")
    print(f"Top-1 delta: {top1_delta:+.6f}")
    print(f"RAM: {memory_gib:.2f}")


if __name__ == "__main__":
    main()
