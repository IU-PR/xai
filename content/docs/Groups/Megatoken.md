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
use the zeroth element of each token’s vector as a signal of importance.
To binarize the decision, we apply a sigmoid function to this value.
The result is a score from 0 to 1, denoted as {{<katex>}}\alpha_i{{</katex>}}, which determines whether a token should
be preserved:

{{<katex display>}} \alpha_i = \sigma\left(\frac{E_i[0]}{\tau} + \beta\right) {{</katex>}}

Here, {{<katex>}}E_i[0]{{</katex>}} is the zeroth element of the token vector, {{<katex>}}\tau{{</katex>}} is a
temperature parameter to control sharpness, and {{<katex>}}\beta{{</katex>}} is a bias term.
Positive bias term is used to shift the decisions towards 1, meaning more tokens are preserved.
This is necessary to stabilize the training, as initially the model might eliminate too many tokens.
