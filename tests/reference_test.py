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


def test_run_reference():
    # vocab_size=10: uniform probability = 1/10
    # dims=4: toy model
    model = utils.UniformLogitModel(vocab_size=10, dims=4)
    tokenizer = utils.FixedTokenizer()

    prompt_text = "a b c d"
    max_tokens = 4

    with unittest.mock.patch("mlx_eval.reference.mlx_lm.load") as mock_load:
        mock_load.return_value = (model, tokenizer)

        result = mlx_eval.reference.run_reference(
            ref_model_path="dummy",
            prompt_text=prompt_text,
            max_tokens=max_tokens,
        )

    assert result["prompt"].tolist() == EXPECTED_PROMPT
    assert mlx.core.allclose(result["log_probs"], EXPECTED_LOGPROBS, atol=1e-6)
    assert result["ppl_mean"] == pytest.approx(EXPECTED_PPL, abs=1e-6)
