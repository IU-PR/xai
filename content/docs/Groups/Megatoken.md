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
At the heart of these breakthroughs lies a powerful concept known as attention — a mechanism that allows models to focus
on the most relevant parts of the input when making predictions.

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

However, there's an inefficiency here.
After enriching the `cat` vector with information from `green`, we still retain a separate vector for `green` — even
though its information has already been incorporated.
This introduces redundancy in the representation.

In this post, we propose a modification to the traditional attention mechanism that reduces such redundancy — preserving
all essential information while removing unnecessary repetition.
