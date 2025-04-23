---
weight: 1
bookFlatSection: true
title: "Megatoken"
---

<style> .markdown a{text-decoration: underline !important;} </style>
<style> .markdown p{text-align: justify} </style>
<style> .markdown h2{font-weight: bold;} </style>

# Megatoken

## Introduction

Ever wonder how models like ChatGPT or image captioning systems understand and generate language so effectively?
At the heart of these breakthroughs lies a powerful concept known as attention.

Before diving into attention, let's briefly revisit the foundation: neural networks.
At their core, neural networks are mathematical functions that transform input data {{<katex>}}x{{</katex>}} into output
values {{<katex>}}y{{</katex>}} through a series of linear and non-linear operations:

{{<katex display>}}y = f(x){{</katex>}}

To use neural networks effectively, we must first convert real-world data — such as text, images, or sound — into
meaningful numerical representations.
In natural language processing, this typically involves splitting a sentence into smaller units called tokens (usually
words or subwords), and representing each token as a vector — a list of numbers that captures some aspects of token's
meaning.

But understanding language isn't just about recognizing individual words. Consider the sentence:

    green cat sat on the mat

Initially, we represent each word as a vector independently using a lookup table, like an embedding dictionary.
At this point, the vector for `cat` doesn't include any information about its color.
That's where the attention mechanism comes in.

The attention mechanism improves token representations by letting each word "attend to" others in the sentence — mixing
in relevant context.
For instance, we might update the vector for `cat` by blending in information from `green`, allowing the model to
capture the idea of a green cat rather than a generic one.

This process is repeated through multiple layers of self-attention, each time refining the representations of tokens by
incorporating more context.

<div style="width: 50%; margin: auto;">
    <img src="/Megatoken/attention.png" alt="Self-Attention"/>
</div>

However, there might be an inefficiency here.
After enriching the `cat` vector with information from `green`, we still retain a separate vector for `green` — even
though its information might have already been incorporated.
This introduces redundancy in the representation.

In this post, we propose a modification to the traditional attention mechanism that reduces such redundancy — preserving
all essential information while removing unnecessary repetition.

<div style="width: 50%; margin: auto;">
    <img src="/Megatoken/comp_attention.png" alt="Compressional Self-Attention"/>
</div>

## Learnable Token Selection

Unlike similar studies, using methods that rely on rigid rules or heuristic thresholds, we adopt a learnable criterion
for token elimination — one that evolves with training and adapts to each input.
The key idea is inspired by the KV-cache optimization technique proposed by NVIDIA:
use the zeroth element of each token's vector as a signal of importance.

To convert this signal into a binary decision, we apply a sigmoid function to this value.
This produces a score between 0 and 1, denoted as {{<katex>}}\alpha_i{{</katex>}}, which determines whether a token
should be preserved:

{{<katex display>}} \alpha_i = \sigma\left(\frac{E_i[0]}{\tau} + \beta\right) {{</katex>}}

Here, {{<katex>}}E_i[0]{{</katex>}} is the zeroth element of the token vector, {{<katex>}}\tau{{</katex>}} is a
temperature parameter to control sharpness, and {{<katex>}}\beta{{</katex>}} is a bias term.
A positive bias shifts the decisions towards 1, encouraging the model to preserve more tokens — a necessary measure to
stabilize early training, when the model might discard too many tokens.

## Differentiable Masking

Although we could completely remove the unwanted tokens from the tensor during training, we used an attention mask to
achieve the same effect — and there's a good reason for that.
To understand why, let's take a closer look at how attention mechanisms work.

In transformer models, attention scores determine how much focus each token should give to every other token in a
sequence.
These scores are stored in a square matrix.
Returning to the earlier example with `green` and `cat`, a high attention score in the row for `cat` and the column for
`green` indicates that `green` strongly influences the updated representation of `cat`.

TODO: EXAMPLE

These scores are computed using the dot product of query and key vectors:
{{<katex display>}} A = \text{softmax} \left( QK^T \right) {{</katex>}}

More explicitly:

{{<katex display>}}
\begin{pmatrix}
q_1 \cdot k_1 & q_1 \cdot k_2 & \cdots & q_1 \cdot k_N \\
q_2 \cdot k_1 & q_2 \cdot k_2 & \cdots & q_2 \cdot k_N \\
\vdots & \vdots & \ddots & \vdots \\
q_N \cdot k_1 & q_N \cdot k_2 & \cdots & q_N \cdot k_N
\end{pmatrix}
{{</katex>}}

Here, query (Q) and key (K) vectors are learned projections of the token embeddings, typically via fully connected
layers.

The attention mask is used to modify these scores.
It allows us to selectively ignore certain tokens by setting their corresponding attention scores to very negative
values (often negative infinity).
After applying the softmax, these scores become effectively zero — ensuring that the model pays no attention to the
masked tokens during that step.

Summing up, to remove the token completely, we can:

1. Set the column entries of pruned tokens to {{<katex>}}-\infty{{</katex>}} so they can't influence other tokens.
2. Set the row entries to {{<katex>}}-\infty{{</katex>}} so they don't get updated by others.
3. Keep the diagonal entry at 0 to allow self-reference and maintain numerical stability in softmax.

For example, to eliminate the first token, the attention mask will look like:

{{<katex display>}}
\begin{pmatrix}
0 & -\inf & -\inf & -\inf \\
-\inf & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
-\inf & 0 & \cdots & 0
\end{pmatrix}
{{</katex>}}

Basically, token selection might be seen as the function, which produces the attention mask for each token:

{{<katex display>}}
M_{j, k}(\alpha_i) =
\begin{cases}
g(\alpha_i), & \text{if } j = i \oplus k = i \\
0, & \text{otherwise}
\end{cases}
{{</katex>}}

where:

{{<katex display>}}
g(\alpha_i) = \begin{cases}
-\inf, & \text{if } \alpha_i \lt 0.5 \\
0, & \text{otherwise}
\end{cases}
{{</katex>}}

Final attention mask might be found as the sum of all attention masks for each token.

{{<katex display>}}
M_{j, k} = \sum_{i=0}^{N}{M_{j, k}(\alpha_i)}
{{</katex>}}

Careful reader might have noticed that learnable criteria must be differentiable.
However, we have a non-continuous function.
Thus, it is not differentiable:

{{<katex display>}}
\lim_{\alpha_i \to 0.5}g(\alpha_i) = -\inf \\
g(0.5) = 0
{{</katex>}}

So complete removal of tokens won't allow criteria training.
We need to allow intermediate states for tokens.
Instead of completely removing the token, we will make it less likely to update others and to be updated.
So we can use {{<katex>}}\ln(\alpha_i){{</katex>}} instead of conditionally defined
{{<katex>}}g(\alpha_i){{</katex>}}.

