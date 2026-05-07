import mlx.core
import mlx.nn


# minimal transformer-like model that outputs uniform logits (equal probability for all tokens)
class UniformLogitModel(mlx.nn.Module):
    # vocab_size: number of possible tokens
    # dims: embedding dimensions
    def __init__(self, vocab_size, dims):
        super().__init__()
        # token embedding layer: maps token ID (0..vocab_size-1) to a dense vector of size embed_dim
        self.embed = mlx.nn.Embedding(vocab_size, dims)
        # language modelling head: maps the embedding vector back to logits over the vocabulary
        self.lm_head = mlx.nn.Linear(dims, vocab_size)
        # set all trainable parameters to zero
        # embedding weights become zero: output zero vectors
        # linear layer weights and bias become zero: logits are all zero
        # result: softmax(logits) gives uniform probability (1/vocab_size) for every token
        self.embed.weight = mlx.core.zeros_like(self.embed.weight)
        self.lm_head.weight = mlx.core.zeros_like(self.lm_head.weight)
        self.lm_head.bias = mlx.core.zeros_like(self.lm_head.bias)

    # x: tensor of token IDs (batch, seq_len)
    def __call__(self, x):
        # (batch, seq_len, embed_dim)
        x = self.embed(x)
        # (batch, seq_len, vocab_size) - raw logits
        return self.lm_head(x)


# model that produces different logits for each token position (non-uniform distribution)
class PositionDependentModel(mlx.nn.Module):
    # vocab_size: number of possible tokens
    # dims: embedding dimensions
    def __init__(self, vocab_size, dims, base_perturbation):
        super().__init__()
        # token embedding layer: maps token ID (0..vocab_size-1) to a dense vector of size embed_dim
        self.embed = mlx.nn.Embedding(vocab_size, dims)
        # language modelling head: maps the embedding vector back to logits over the vocabulary
        self.lm_head = mlx.nn.Linear(dims, vocab_size)
        # set all trainable parameters to zero
        # embedding weights become zero: output zero vectors
        # linear layer weights become zero: logits reduce to bias only
        self.embed.weight = mlx.core.zeros_like(self.embed.weight)
        self.lm_head.weight = mlx.core.zeros_like(self.lm_head.weight)
        # perturbation matrix: bias[i][i] = base_perturbation * (i+1)
        biases = mlx.core.zeros((vocab_size, vocab_size))
        for i in range(vocab_size):
            biases = biases.at[i, i].add(base_perturbation * (i + 1))
        # store the bias matrix as a persistent buffer (not a trainable parameter)
        self.bias_matrix = biases

    # x: tensor of token IDs (batch, seq_len)
    def __call__(self, x):
        # remove batch dimension (seq_len,)
        token_ids = x.squeeze(0)
        # each token ID gets its own bias row
        logits = self.bias_matrix[token_ids]
        # add batch dimension back (1, seq_len, vocab_size)
        return logits[None, :, :]
