# Counterfactual Explanations for Credit Risk Models: A Case Study

---

## TL;DR

In this case study, we implement and incrementally refine a gradient-based approach for generating **counterfactual explanations** in credit risk modeling. Beginning with a basic optimization procedure, we identify and resolve multiple real-world issues:

- Immutable and semantically constrained features
- Inter-feature dependencies (e.g., derived ratios)
- Ordinal variables requiring discrete treatment
- One-hot encoded categorical features demanding joint behavior

We develop solutions involving **gradient masking**, **differentiable approximations**, **manual feature injection**, and the **Gumbel-Softmax trick** to ensure our counterfactuals are effective, valid, interpretable, and realistic.

---

## 1. Introduction: What Are Counterfactual Explanations?

Given a binary classifier {{<katex>}} f(\mathbf{x}) {{</katex>}} trained to predict whether a loan applicant will **default**, counterfactual explanations aim to answer:

> *“What minimal changes to the applicant’s features would have changed the model’s decision?”*

This is framed as an optimization:

{{<katex display>}}
\min_{\mathbf{x}'} \; d(\mathbf{x}, \mathbf{x}') + \lambda \,\mathcal{L}(f(\mathbf{x}'), c)
{{</katex>}}

Where:
- {{<katex>}} \mathbf{x} {{</katex>}}: original input (e.g., borrower profile)
- {{<katex>}} \mathbf{x}' {{</katex>}}: counterfactual
- {{<katex>}} d(\cdot) {{</katex>}}: distance measure (e.g., {{<katex>}} L_2 {{</katex>}})
- {{<katex>}} \mathcal{L} {{</katex>}}: classification loss (e.g., binary cross-entropy)
- {{<katex>}} c {{</katex>}}: desired target label (e.g., “non-default”)
- {{<katex>}} \lambda {{</katex>}}: hyperparameter for trade-off

This setup seeks small yet decisive modifications leading to a different prediction.

---

## 2. Feature Overview and Data Constraints

[Unchanged content...]

---

## 3. Baseline Counterfactual Optimization

[Unchanged content...]

---

## 4. Maintaining Derived Feature Consistency

### Problem

We expected:

{{<katex display>}}
\text{loan\_percent\_income} = \frac{\text{loan\_amnt}}{\text{person\_income}}
{{</katex>}}

[Unchanged content...]

---

## 5. Treating Ordinal Features Correctly

### Solutions

#### A. Snapping

After each step:

{{<katex display>}}
x_{\text{ordinal}} = \text{round}(x_{\text{ordinal}})
{{</katex>}}

#### B. Soft-Round Penalty

We added a soft penalty to the loss:

{{<katex display>}}
\alpha \sum_{i \in \text{ordinals}} \left| x_i - \text{soft\_round}(x_i) \right|
{{</katex>}}

Where:

{{<katex display>}}
\text{soft\_round}(x) = \lfloor x \rfloor + \sigma\left( \beta(x - \lfloor x \rfloor - 0.5) \right)
{{</katex>}}

[Unchanged content...]

---

## 6. Ensuring Valid One-Hot Encoded Categorical Features

### Desired Behavior

If one value in a one-hot group is increased, all others should go to 0:

{{<katex display>}}
\sum_{j=1}^K x_j = 1, \quad x_j \in \{0, 1\}
{{</katex>}}

### Solution: Gumbel-Softmax Trick

Each one-hot group is represented as:

{{<katex display>}}
\mathbf{y} = \text{softmax}\left( \frac{\log \boldsymbol{\pi} + \mathbf{g}}{\tau} \right)
{{</katex>}}

Where:
- {{<katex>}} \mathbf{g} \sim \text{Gumbel}(0, 1) {{</katex>}}
- {{<katex>}} \tau {{</katex>}}: temperature parameter

[Unchanged content...]

---

## 7. Full Optimization Pipeline

[Unchanged content...]

---

## 8. Summary of Techniques

[Unchanged content...]

---

## 9. Technical Implementation

[Unchanged content...]

---

## 10. Conclusion

[Unchanged content...]

---

## References

[Unchanged content...]

