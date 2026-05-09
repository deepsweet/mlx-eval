import gc
import math
import pathlib
import statistics
import sys

import mlx.core
import mlx_lm

from . import utils


def run_compare(
    model,
    prompt,
    ref_log_probs,
    ref_ppl_mean,
    ref_top1_acc,
):
    """
    Run a target model comparison using reference log-probabilities,
    and return KL divergence, perplexity, and top-1 accuracy metrics.
    """

    result = utils.run_model(model, prompt)
    log_probs = result["log_probs"]

    # per-token KL divergence summed over vocabulary (batch_size, max_tokens)
    kld_none = mlx.nn.losses.kl_div_loss(log_probs, ref_log_probs, reduction="none")
    kld_mean = mlx.core.mean(kld_none).item()
    kld_list = kld_none.flatten().tolist()
    kld_p95 = statistics.quantiles(kld_list, n=100)[-5]
    kld_p99 = statistics.quantiles(kld_list, n=100)[-1]

    ppl_delta = result["ppl_mean"] - ref_ppl_mean
    top1_delta = result["top1_acc"] - ref_top1_acc

    # now free log_probs
    del log_probs, kld_none

    return {
        "kld_mean": kld_mean,
        "kld_list": kld_list,
        "kld_p95": kld_p95,
        "kld_p99": kld_p99,
        "ppl_mean": result["ppl_mean"],
        "ppl_delta": ppl_delta,
        "top1_acc": result["top1_acc"],
        "top1_delta": top1_delta,
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

    # load model once, reuse across all windows
    mlx.core.clear_cache()
    memory_before = mlx.core.get_active_memory()
    model = mlx_lm.utils.load_model(target_model_path)[0]
    memory_after = mlx.core.get_active_memory()
    memory = memory_after - memory_before

    output_dir = pathlib.Path("outputs")

    for i in range(window_count):
        window_num = i + 1
        window_file = output_dir / f"{window_num:02d}.npz"

        print(f"\nProcessing window {window_num}/{window_count}")

        ref_data = mlx.core.load(window_file)
        prompt = ref_data["prompt"]
        ref_log_probs = ref_data["log_probs"]
        ref_ppl_mean = ref_data["ppl_mean"].item()
        ref_top1_acc = ref_data["top1_acc"].item()

        result = run_compare(
            model,
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

        print(f"KLD: {result['kld_mean']:.6f}")
        print(f"KLD p95: {result['kld_p95']:.6f}")
        print(f"KLD p99: {result['kld_p99']:.6f}")
        print(f"PPL: {result['ppl_mean']:.6f}")
        print(f"Δ PPL: {result['ppl_delta']:+.6f}")
        print(f"Acc@1: {result['top1_acc']:.6f}")
        print(f"Δ Acc@1: {result['top1_delta']:+.6f}")

        if i == 0:
            memory_gib = memory / (1024**3)

        del ref_data, prompt, ref_log_probs, result
        gc.collect()
        mlx.core.clear_cache()

    del model
    gc.collect()
    mlx.core.clear_cache()

    # overall token-weighted average metrics
    overall_ppl = math.exp(log_ppl_weighted / total_pred)
    overall_top1 = correct_weighted / total_pred
    overall_kld_mean = kld_weighted / total_pred

    # pool all per-token KLD values for exact overall percentiles
    overall_kld_p95 = statistics.quantiles(all_kld, n=100)[-5]
    overall_kld_p99 = statistics.quantiles(all_kld, n=100)[-1]

    overall_file = output_dir / "overall.npz"
    ref_overall = mlx.core.load(overall_file)
    ref_ppl_overall = ref_overall["ppl_mean"].item()
    ref_top1_overall = ref_overall["top1_acc"].item()

    ppl_delta = overall_ppl - ref_ppl_overall
    top1_delta = overall_top1 - ref_top1_overall

    print(f"\nKLD: {overall_kld_mean:.6f}")
    print(f"KLD p95: {overall_kld_p95:.6f}")
    print(f"KLD p99: {overall_kld_p99:.6f}")
    print(f"PPL: {overall_ppl:.6f}")
    print(f"Δ PPL: {ppl_delta:+.6f}")
    print(f"Acc@1: {overall_top1:.6f}")
    print(f"Δ Acc@1: {top1_delta:+.6f}")
    print(f"RAM: {memory_gib:.2f}")


if __name__ == "__main__":
    main()
