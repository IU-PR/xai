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

Ever wonder how models like ChatGPT understand and generate language so well?
A big part of the answer is attention — a mechanism that allows each word to adjust its meaning based on its context in
a sentence.

Let's start with a simple example:

> green cat sat on the mat

At first, each word is turned into a standalone vector — a numerical representation that captures its general meaning.
But at this stage, the vector for `cat` doesn't "know" that it's being described as `green`.

That's where attention comes in.
It lets each word "look at" others in the sentence and pull in useful context.
For instance, attention can update the vector for `cat` by blending in information from `green`, helping the model
understand it's not just a `cat` — it's a `green cat`.

This context-mixing happens multiple times, each time sharpening the model's understanding of the sentence.

<div style="width: 50%; margin: auto;">
    <img src="/Megatoken/attention.png" alt="Self-Attention"/>
</div>

But here's a question: once `cat` has absorbed info from `green`, do we really need to keep the `green` vector around?
Chances are, its meaning has already been passed on.
Keeping both creates redundancy — extra baggage the model has to carry.

**Megatoken** is about reducing that baggage.
We introduce a method that learns to drop tokens that are no longer adding value — keeping only what's truly important.

## Learnable Token Selection

Some earlier approaches use fixed rules to decide which tokens to eliminate to produce a single output vector.
Instead, we use a learnable method — one that adapts to each input and dynamically decides how many tokens to keep.

Here's the key idea: for each token, we examine the first value in its embedding vector as a kind of importance signal.
We then pass this value through a sigmoid function to get a score between 0 and 1.
This score tells us how likely it is that a token should be kept in the computation.

{{<katex display>}} \alpha_i = \sigma\left(\frac{E_i[0]}{\tau} + \beta\right) {{</katex>}}

Where:

- {{<katex>}}E_i[0]{{</katex>}} is the first element of the token's embedding,
- {{<katex>}}\tau{{</katex>}} is a temperature parameter, which controls how sharp the selection is,
- {{<katex>}}\beta{{</katex>}} is a bias term that helps preserve more tokens early in training.

## Differentiable Masking

So how do we actually remove a token from attention?

We can't simply delete it from the tensor.
Instead, we use a mask — a special matrix that tells the model which tokens to ignore.
You'll see why this is necessary in a moment.

In transformer models, attention is computed as:

{{<katex display>}} A = \text{softmax} \left( QK^T + M \right) {{</katex>}}

Each value in this matrix indicates how much one token attends to another.
To remove a token from attention, we need to ensure its scores are effectively zeroed out.
This is done using an attention mask {{<katex>}}M{{</katex>}}, which is constructed as follows:

1. Set token's **column** to {{<katex>}}-\infty{{</katex>}}, so other tokens do not attend to it,
2. Set token's **row** to {{<katex>}}-\infty{{</katex>}}, so it does not attend to others,
3. Set **diagonal** to 0, allowing the token to reference itself, which helps maintain numerical stability during
   softmax.

For example, to remove the first token, the attention mask would look like:

{{<katex display>}}
M = \begin{pmatrix}
0 & -\inf & -\inf & -\inf \\
-\inf & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
-\inf & 0 & \cdots & 0
\end{pmatrix}
{{</katex>}}

To create such masks, we define a function that depends on each token's importance score
{{<katex>}}\alpha_i{{</katex>}}:

{{<katex display>}}
M_{j, k}(\alpha_i) =
\begin{cases}
g(\alpha_i), & \text{if } j = i \oplus k = i \\
0, & \text{otherwise}
\end{cases}
{{</katex>}}

At first glance, it might seem natural to define {{<katex>}}g(\alpha_i){{</katex>}} with a hard threshold to achieve the
same effect as removing token from tensor:

{{<katex display>}}
g(\alpha_i) = \begin{cases}
-\inf, & \alpha_i < 0.5 \\
0, & \text{otherwise}
\end{cases}
{{</katex>}}

But this kind of step function isn't differentiable — it has zero gradients everywhere, except an undefined gradient at
jump point 0.5.
So complete elimination isn't suitable for training with gradient-based optimization.

To fix this, we use a smooth approximation based on the natural logarithm:

{{<katex display>}}
g(\alpha_i) = \ln(\alpha_i)
{{</katex>}}

This way, we don't remove tokens completely — we just gradually reduce their influence depending on how low
their {{<katex>}}\alpha_i{{</katex>}} score is.
This approach preserves differentiability and allows the model to learn which tokens matter most.