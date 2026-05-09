import gc
import math

import mlx.core
import mlx.nn


def run_model(model, prompt):
    # raw logits per token from forward pass over vocabulary
    logits = model(prompt)
    # materialise logits, break giant lazy graph
    mlx.core.eval(logits)
    # convert logits to numerically stable log-probabilities along the vocabulary axis
    log_probs = mlx.nn.log_softmax(logits, axis=-1)
    # materialise logprobs, detach from logits
    mlx.core.eval(log_probs)

    # cleanup logits
    del logits
    gc.collect()
    mlx.core.clear_cache()

    # drop last token because there is no next token to predict
    shift_log_probs = log_probs[:, :-1, :]
    # drop first token because there is no previous token to use as context for prediction
    shift_prompt = prompt[:, 1:]
    # add a trailing dimension to match the shape expected by take_along_axis
    gather_idx = shift_prompt[..., None]
    # log-probabilities of the true next tokens
    target_log_probs = mlx.core.take_along_axis(shift_log_probs, gather_idx, axis=-1)
    # average log-probability of the correct token over all positions
    mean_log_prob = mlx.core.mean(target_log_probs)
    # cross-entropy = negative mean log-probability
    cross_entropy = -mean_log_prob.item()
    # convert cross-entropy to perplexity
    ppl_mean = math.exp(cross_entropy)

    # top-1 accuracy: fraction of tokens where the predicted token matches the true next token
    top1_preds = mlx.core.argmax(shift_log_probs, axis=-1)
    top1_acc = mlx.core.mean(top1_preds == shift_prompt).item()

    del shift_log_probs, shift_prompt, top1_preds

    return {
        "log_probs": log_probs,
        "ppl_mean": ppl_mean,
        "top1_acc": top1_acc,
    }
