---
weight: 1
bookFlatSection: true
title: "Interpretable SHAP for Credit Risk Scoring"
---

<style> .markdown a{text-decoration: underline !important;} </style>
<style> .markdown h2{font-weight: bold;} </style>

# Interpretable SHAP for Credit Risk Scoring

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Application Domain](#2-application-domain)
- [3. Methodology](#3-methodology)
  - [3.1 Dataset and Model Overview](#31-dataset-and-model-overview)
  - [3.2 Explainability Method: SHAP](#32-explainability-method-shap)
- [4. Implementation: Applying SHAP to RandomForest](#4-implementation-applying-shap-to-randomforest)
  - [4.1 Implementation Overview](#41-implementation-overview)
  - [4.2 How to Use It](#42-how-to-use-it)
- [5. Experiments and Analysis](#5-experiments-and-analysis)
  - [5.1 Evaluation Metrics](#51-evaluation-metrics)
  - [5.2 Explanation Techniques and Visualizations](#52-explanation-techniques-and-visualizations)
  - [5.3 Interpretations and Findings](#53-interpretations-and-findings)
- [6. My Implementation](#6-my-implementation)
- [7. Conclusion, Future Work, Limitations and Ethical Considerations](#7-conclusion-future-work-limitations-and-ethical-considerations)
- [8. References](#8-references)

---

## 1. Introduction

This project explores how SHAP (SHapley Additive exPlanations) can be used to enhance interpretability in credit scoring. The core objective is to implement SHAP from scratch and apply it to a Random Forest model trained on a credit dataset.

**Research Question:** *Can we accurately replicate SHAP value explanations without relying on libraries, and how useful are these explanations for understanding credit risk decisions made by machine learning models?*

SHAP was chosen due to its strong theoretical grounding in cooperative game theory and its ability to produce consistent, locally accurate feature attributions. This makes it particularly suitable for credit scoring applications where decisions have significant human impact.

By replicating SHAP manually, we aim to:

- Gain deeper insight into the mechanics of explainable AI
- Validate model decisions through transparent breakdowns
- Provide stakeholders and regulators with interpretable model behavior

This research bridges the gap between theoretical fairness requirements and practical implementation in real-world credit assessment systems.


---

## 2. Application Domain

Credit scoring is a critical application within financial services, where institutions assess an applicant's ability to repay a loan. An accurate and interpretable model helps mitigate financial risk and ensures transparency in lending decisions.

By applying SHAP to a Random Forest model on the German Credit dataset, this project addresses the need for explainability in high-stakes domains. Regulatory frameworks (like the EU GDPR or U.S. Equal Credit Opportunity Act) increasingly require that decisions be interpretable and non-discriminatory. Hence, this research not only contributes technically but also aligns with real-world financial compliance needs.

In this application domain, local explanations can help loan officers understand individual decisions, while global explanations support model auditability and fairness assessments.



---

## 3. Methodology

### 3.1 Dataset and Model Overview

We used the [**German Credit Dataset**](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)), a structured dataset frequently applied in credit risk modeling research. It contains 1000 samples with 20 features and a binary target variable indicating creditworthiness (good or bad credit risk). The features include a mix of categorical and numerical variables such as:

- `checking_status`: status of existing checking account
- `duration`: duration in months
- `credit_history`: credit history records
- `purpose`: purpose of the loan (e.g., car, education)
- `credit_amount`: amount of credit requested
- `savings_status`: status of savings account/bonds
- `employment`: years of current employment
- `installment_commitment`: installment rate as a percentage of income
- `personal_status`: personal and sex status
- `other_parties`: other debtors/guarantors
- `residence_since`: years living at current residence
- `property_magnitude`: value of assets
- `age`: applicant's age
- `other_payment_plans`: presence of other installment plans
- `housing`: housing situation
- `existing_credits`: number of existing credits
- `job`: type of job
- `num_dependents`: number of dependents
- `own_telephone`: ownership of a telephone
- `foreign_worker`: foreign worker status

These features offer a comprehensive view of an individual's financial and personal profile relevant to credit decision-making.

We selected the **Random Forest classifier** due to its ensemble-based structure that combines multiple decision trees to improve predictive performance and reduce overfitting. Random Forest is particularly suitable for tabular data with mixed data types (categorical and numerical), and it performs implicit feature selection by evaluating feature importance across many trees. Each tree in the forest is trained on a bootstrap sample of the data and considers a random subset of features at each split, which enhances diversity and robustness. Furthermore, Random Forest supports probability estimation and has interpretable decision paths, making it a practical and explainable choice for credit risk modeling.

**Why Random Forest?**

- **Robust performance on tabular data** — It effectively handles both numerical and categorical variables.
- **Resistance to overfitting** — Uses bootstrap aggregation and feature randomness.
- **Feature importance** — Naturally provides importance rankings useful for interpretability.
- **Interpretability** — Easier to interpret than black-box models like neural networks.
- **Practicality** — Widely supported, scalable, and requires minimal hyperparameter tuning.

### 3.2 Explainability Method: SHAP

SHAP (SHapley Additive exPlanations) is an explainability framework based on Shapley values from cooperative game theory. It provides consistent and locally accurate attributions of a model’s output to each input feature.

In SHAP, the prediction task is treated as a game in which the "players" are the input features. The goal is to fairly allocate the model output (gain) among the features based on their contribution to the prediction.

#### How SHAP Works

- It considers all possible feature combinations (subsets) and computes the marginal contribution of a feature across these combinations.
- Each SHAP value represents how much a feature contributes to the difference between the actual prediction and the mean prediction (baseline).
- The final explanation is a sum of these SHAP values and the base value.

Mathematically, for a model \(f\), the SHAP value for feature \(i\) is:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]
$$

Where:

- \(F\): the set of all features
- \(S\): a subset of features excluding \(i\)

This formulation ensures fairness (symmetry), efficiency (conservation of output), and consistency across feature attributions.

SHAP is model-agnostic and provides a unified measure of feature importance for both global and local interpretability.

### 3.3 Kernel SHAP Approximation

Kernel SHAP is a model-agnostic approximation algorithm that estimates SHAP values using a weighted linear regression. It is especially useful when the exact computation of Shapley values is computationally infeasible, such as in models with many input features.

Key aspects of Kernel SHAP:

- It treats the original model as a black box.
- It samples feature subsets and queries the model to observe how predictions change.
- It solves a weighted least squares problem to estimate each feature's SHAP value.
- Weights are chosen such that smaller subsets (i.e., those missing more features) have higher influence.

Kernel SHAP combines the strengths of LIME (local surrogate models) and Shapley theory, and provides theoretically grounded explanations even when the model internals are not accessible.

Since computing all {{<katex>}}( 2^n ){{</katex>}} feature coalitions is infeasible for large {{<katex>}}( n ){{</katex>}}, Kernel SHAP uses a model-agnostic, sampling-based approach that approximates SHAP values via weighted linear regression.

**Steps:**
1. Generate random binary masks {{<katex>}}( z \in {0, 1}^n ){{</katex>}}
2. For each mask, compute a masked input instance:
   {{<katex>}}[ x_z = z \odot x + (1 - z) \odot \bar{x}_{bg} ]{{</katex>}}
3. Evaluate the model output {{<katex>}}( f(x_z) ){{</katex>}}
4. Solve:
   {{<katex>}}[ \hat{f}(z) = \phi_0 + \sum_{i=1}^n \phi_i z_i ]{{</katex>}}
   using weighted ridge regression, where the weights are given by the SHAP kernel:
   {{<katex>}}[ w(z) = \frac{(n - 1)}{{n \choose |z|} \cdot |z|(n - |z|)} ]{{</katex>}}

This formulation ensures that explanations are consistent and robust, even when internal model logic is inaccessible.

---

## 4. Implementation: Applying SHAP to RandomForest

### 4.1 Implementation Overview

Our custom SHAP implementation includes:

- **Expected value** computation: model prediction averaged over the training data.
- **Marginal contribution** estimation: for each feature, we average the effect across multiple permutations.
- **SHAP value aggregation**: computes a final breakdown per instance.

This was done without using the `shap` package.


Implementing SHAP from scratch for a Random Forest model presented several practical challenges. Kernel SHAP assumes feature independence and requires repeated sampling and model evaluations, making it computationally expensive. Additionally, since Random Forests are not inherently differentiable or linear, accurate approximation depends on well-designed masking strategies and sufficient sampling.

To address these, we:

- Used a **mean-masked baseline** over multiple background samples to better approximate realistic counterfactuals.
- Tuned the **SHAP kernel weights** and **regularization strength** in ridge regression to improve numerical stability.
- Increased the number of sampled coalitions (up to 2000) to improve approximation accuracy.

These design choices helped us replicate SHAP explanations with high fidelity, while keeping the implementation self-contained and interpretable.

### 4.2 How to Use It

The notebook:

1. Trains a Random Forest model.
2. Applies SHAP using our method.
3. Visualizes outputs using:
   - Custom force plots (HTML with SVG)
   - Waterfall charts
   - Top-K bar plots
   - SHAP summary scatter plots

---
### 4.3 SHAP Implementation Steps with Code Snippets

1. **Sample feature masks:**
```python
masks = np.random.randint(0, 2, size=(nsamples, n_features))
```

2. **Mask inputs and generate predictions:**
```python
for mask in masks:
    x_masked = np.where(mask == 1, x, X_bg.mean(axis=0))
    pred = model.predict_proba([x_masked])[0][1]
```

3. **Compute kernel weights:**
```python
def shap_kernel_weight(mask):
    z = mask.sum()
    n = len(mask)
    return (n - 1) / (comb(n, z) * z * (n - z)) if 0 < z < n else 1e-6
```

4. **Fit weighted ridge regression:**
```python
reg = Ridge(alpha=1e-3)
reg.fit(X_masked, preds, sample_weight=weights)
shap_values = reg.coef_
```

5. **Return SHAP values and base value:**
```python
base_value = reg.intercept_
return shap_values, base_value
```

---

## 5. Results and Visualizations

### 5.1 Explanation Techniques and Visualizations

To understand both individual predictions and overall model behavior, we employed a combination of **local** and **global** explanation methods using SHAP.

- **Local explanations** (force plots, bar plots, waterfall charts) clarify how specific input features influenced the outcome for one applicant.
- **Global explanations** (summary plots, interaction matrices) reveal patterns across the entire dataset, highlighting which features are most impactful and how they interact.

These complementary techniques allow for both micro-level decision inspection and macro-level model transparency, making them essential in domains such as credit scoring where interpretability is a regulatory and ethical necessity.

We generated 11 distinct SHAP visualizations including:

- **Local explanation**: force plots, bar plots, and waterfall charts for individual predictions
- **Global explanation**: summary plots, interaction matrices, and comparison between risky and safe applicants

### 5.2 Interpretations and Findings

**Summary of key insights from visualizations:**

- `checking_status` consistently appears as the most influential feature in both local and global plots.
- `age`, `duration`, and `credit_history` are strong negative contributors for younger applicants or long loans.
- Positive factors such as `purpose`, `property_magnitude`, and `employment` help increase predicted creditworthiness.
- Interaction visualizations reveal that features like `checking_status` and `duration` compound their effects in risky applicants.
- Our custom SHAP implementation approximates SHAP values with high fidelity (mean absolute difference \~0.017 vs KernelExplainer).

### Visual Explanations in detatils

### 1. SHAP Value Bar Plot Interpretation (One Prediction)
![Fig. 1 SHAP Value Bar Plot for 1 Prediction](/SHAP_for_credit_risk/1_Bar_plot_for_1_prediction.png)
- The most influential negative contributors are `checking_status` and `age`, indicating perceived financial and demographic risks.
- Positive contributions from `purpose`, `installment_commitment`, and `residence_since` help balance the prediction.

### 2. SHAP Force Plot Interpretation
[Link to the SHAP Force Plot](https://github.com/Diana-Vostrova/xai/blob/master/static/SHAP_for_credit_risk/force_plot_custom.html)
- The force plot shows how the model moves from the average output to the final prediction based on individual features.
- Key drivers include `credit_history`, `savings_status`, and `checking_status`.

### 3. Interpretation of SHAP Bar Plot – Top 10 Features
![Fig. 3 SHAP Bar Plot – Top 10 Features](/SHAP_for_credit_risk/3_Top_10_Important_Features_for_Prediction.png)
- `checking_status`, `age`, and `personal_status` are the top negative contributors.
- Features like `purpose`, `installment_commitment`, and `property_magnitude` help raise the prediction.

### 4. SHAP Waterfall Chart Interpretation
![Fig. 4 SHAP Waterfall Chart](/SHAP_for_credit_risk/4_SHAP_Waterfall_Chart.png)
- The prediction decreases from the base value due to strong negative SHAP values for `checking_status` and `age`.
- Positive contributions include `purpose` and `property_magnitude`, but do not fully offset the negatives.

### 5. SHAP Summary Plot Interpretation
![Fig. 5 SHAP Summary Plot](/SHAP_for_credit_risk/5_Summary_plot.png)
- Globally, `checking_status`, `duration`, and `age` are the most influential features.
- Low values for `age` and high values for `duration` push predictions lower.

### 6. SHAP Interaction Plot: `age` vs SHAP(`age`), colored by `duration`
![Fig. 6 SHAP Interaction Plot](/SHAP_for_credit_risk/6_Interaction_age_by_duration.png)
- Younger age consistently decreases predictions.
- Longer loan durations amplify the negative impact of age.

### 7. SHAP Feature Interaction: `credit_history` × `other_payment_plans`
![Fig. 7 SHAP Feature Interaction: `credit_history` × `other_payment_plans`](/SHAP_for_credit_risk/7_Interaction_credit_history_by_other_payment_plans.png)
- Applicants with no credit history and no other plans are penalized most.
- Good credit history matters more when not undermined by additional obligations.

### 8. SHAP Interaction Value Matrix — Interpretation
![Fig. 8 SHAP Interaction Value Matrix](/SHAP_for_credit_risk/8_Full_SHAP_Interation_Value_matrix.png)
- `checking_status` has strong independent influence.
- Features like `purpose` and `credit_amount` show notable interactions.

### 9. SHAP Interaction Comparison: Risky Client vs Safe Client
![Fig. 9 SHAP Interaction Comparison: Risky Client vs Safe Client](/SHAP_for_credit_risk/9_SHAP_interaction_comparison_risk_vs_safe_client.png)
- Risky clients are more impacted by interactions combining weak financial signals.
- Safe clients benefit from combinations of stable signals.

### 10. SHAP Waterfall Comparison: Risky vs Safe Client
![Fig. 10 SHAP Waterfall Comparison: Risky vs Safe Client](/SHAP_for_credit_risk/10_Waterfall_Comparison_of_SHAP_Contributions_Risky_vs_Safe_client.png)
- Risky client is penalized by multiple negative features with no major positives.
- Safe client gains modest positive contributions and avoids major penalties.

### 11-12. SHAP Value Comparison: Custom vs shap.KernelExplainer
![Fig. 11 SHAP Value Difference: Custom vs shap.KernelExplainer](/SHAP_for_credit_risk/11_SHAP_Value_Difference_Ours_vs_shap.KernelExplainer.png)
![Fig. 12 SHAP Value Comparison: Custom vs shap.KernelExplainer](/SHAP_for_credit_risk/12_SHAP_Value_Comparison.png)
- Our custom SHAP implementation shows strong agreement with KernelExplainer.
- Mean absolute SHAP difference is low, confirming the accuracy of our method.

---

## 6. My Implementation

All code, visualizations, and notebooks are available in the GitHub repository (to be added):

> 📎 **Link to GitHub repo:** **coming soon**\
> Contains code, SHAP implementation logic, and interactive visualizations.

---

## 7. Conclusion

This project demonstrates how manual implementation of SHAP can:

- Make ML model decisions more transparent
- Offer causal-like reasoning for features
- Improve trust and accountability in credit scoring

### Future Work

- Test with other models (e.g., XGBoost, Logistic Regression)
- Add comparative analysis with the SHAP library
- Create an interactive dashboard for model inspection
- Measure and mitigate fairness issues

### Limitations

- Manual SHAP is slower and more fragile than library versions.
- Assumes feature independence — not always realistic.
- Findings may not generalize due to dataset size and sampling.

### Ethical Considerations

- Credit scoring affects real lives — transparency and fairness are essential.
- Explainable models can empower both users and regulators.

---

## 8. References

- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*. [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
- Lundberg, S. M., Erion, G. G., & Lee, S. I. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*. [https://www.nature.com/articles/s42256-019-0138-9](https://www.nature.com/articles/s42256-019-0138-9)
- Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual Explanations without Opening the Black Box. *Harvard Journal of Law & Technology*, 31(2)
- UCI Machine Learning Repository. German Credit Dataset. [https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- SHAP GitHub Repository. [https://github.com/shap/shap](https://github.com/shap/shap)
