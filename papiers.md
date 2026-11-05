# Papers

## [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

DeepSeek-V3 is a 671B-parameter Mixture-of-Experts model that activates 37B
parameters per token, using MLA and multi-token prediction for efficient,
high-quality training and inference.

## [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

Language-model loss improves predictably with more parameters, data, and
training compute, enabling informed scaling decisions.

## [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)

For a fixed compute budget, model parameters and training tokens should scale
together; a practical rule is roughly 20 training tokens per parameter.

## [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871)

Scaling laws for MoE models show that fine-grained experts can outperform dense
Transformers, with their efficiency advantage growing at larger scale.

## [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)

Switch Transformers route each token to one expert, enabling sparse trillion-
parameter models with much lower compute cost than dense models.

## [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/pdf/2309.00071)

YaRN efficiently extends the context window of RoPE-based LLMs, requiring fewer
fine-tuning tokens and training steps than earlier context-extension methods.

## [All the Transformer Math You Need to Know](https://jax-ml.github.io/scaling-book/transformers/#global-flops-and-params-calculation)

Reference for calculating Transformer parameters and training FLOPs, including
the common approximation: `6 × parameters × training tokens`.

## [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

Introduces RoPE: a rotation-based position encoding that incorporates relative
token positions directly into self-attention.
