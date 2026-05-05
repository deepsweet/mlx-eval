import math
import unittest.mock

import mlx.core
import pytest
import utils

import mlx_eval

# fixed sequence of 4 token IDs (batch dimension added later)
EXPECTED_PROMPT = [[0, 1, 2, 3]]
# log-probability of each token under a uniform distribution over vocab_size=10
# -ln(1/10) = ln(10)
LOGPROB_VALUE = -math.log(10)
# expected log-probabilities tensor for the reference model (uniform distribution)
# shape (1, 4, 10): 1 batch, 4 positions, 10 vocabulary entries
EXPECTED_LOGPROBS_ARR = [[[LOGPROB_VALUE] * 10] * 4]
EXPECTED_LOGPROBS = mlx.core.array(EXPECTED_LOGPROBS_ARR)
# expected perplexity of the uniform reference model
# exp(ln(10)) = 10
EXPECTED_PPL = 10
# expected top-1 accuracy of the uniform reference model
# With uniform logits, argmax returns first class (0) for all positions.
# Shifted prompt tokens are [1,2,3] → none match 0.
EXPECTED_TOP1_ACC = 0.0

# Precomputed values for the PositionDependentModel with base_perturbation=0.01
EXPECTED_VARYING_KLD_MEAN = 3.391503923921846e-05
EXPECTED_VARYING_KLD_P95 = 9.675684850662947e-05
EXPECTED_VARYING_KLD_P99 = 0.00010317994747310877
EXPECTED_VARYING_PPL_MEAN = 10.020231246948242
EXPECTED_VARYING_PPL_DELTA = 0.020231246948242188


def test_run_compare_self():
    model = utils.UniformLogitModel(vocab_size=10, dims=4)
    tokenizer = utils.FixedTokenizer()

    with unittest.mock.patch("mlx_eval.compare.mlx_lm.load") as mock_load:
        mock_load.return_value = (model, tokenizer)

        result = mlx_eval.compare.run_compare(
            target_model_path="dummy",
            prompt=mlx.core.array(EXPECTED_PROMPT),
            ref_log_probs=EXPECTED_LOGPROBS,
            ref_ppl_mean=EXPECTED_PPL,
            ref_top1_acc=EXPECTED_TOP1_ACC,
        )

    assert result["kld_mean"] == 0
    assert result["kld_p95"] == 0
    assert result["kld_p99"] == 0
    assert result["ppl_mean"] == pytest.approx(EXPECTED_PPL, abs=1e-6)
    assert result["ppl_delta"] == pytest.approx(0, abs=1e-6)
    assert result["memory"] >= 0


def test_run_compare_varying_perturbation():
    target_model = utils.PositionDependentModel(vocab_size=10, dims=4, base_perturbation=0.01)
    tokenizer = utils.FixedTokenizer()

    with unittest.mock.patch("mlx_eval.compare.mlx_lm.load") as mock_load:
        mock_load.return_value = (target_model, tokenizer)

        result = mlx_eval.compare.run_compare(
            target_model_path="dummy",
            prompt=mlx.core.array(EXPECTED_PROMPT),
            ref_log_probs=EXPECTED_LOGPROBS,
            ref_ppl_mean=EXPECTED_PPL,
            ref_top1_acc=EXPECTED_TOP1_ACC,
        )

    assert result["kld_mean"] == EXPECTED_VARYING_KLD_MEAN
    assert result["kld_p95"] == EXPECTED_VARYING_KLD_P95
    assert result["kld_p99"] == EXPECTED_VARYING_KLD_P99
    assert result["ppl_mean"] == EXPECTED_VARYING_PPL_MEAN
    assert result["ppl_delta"] == EXPECTED_VARYING_PPL_DELTA
    assert result["memory"] >= 0
