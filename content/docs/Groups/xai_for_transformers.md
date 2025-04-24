# XAI for Transformers. Explanations through LRP

## Introduction

Transformers are becoming more and more common these days. But transformers are based on DNN that makes it harder to explain than other models. However, more and more ordinary users are starting to work with LLMs and to have more questions and doubts for its' work and decisions. Thus, there is a need for some explanation of Transformers. The method presented in the article ["XAI for Transformers: Better Explanations through Conservative Propagation"](https://proceedings.mlr.press/v162/ali22a/ali22a.pdf) by Ameen Ali et. al. serves this purpose. 

## LRP method

Layer-wise Relevance Propagation method here are compared with Gradient×Input method presented in [earlier article](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf) in this field.

<p align="center">
    <img src="/xai_for_transformers/1.png" width='300' alt="LRP outlook"/>
</p>
<h5><font color="DimGray"><center>Img.1. Layer-wise Relevance Propagation principe</center></font></h5>

The relevence in LRP is computing as
$$R(x_i) = \sum_{j} \frac{\delta y_j}{\delta x_i} \frac{x_i}{y_j} R(y_j)$$

But in some layers of transformer formulas look little different. For the attention-head layer and for normalization layers rules are look like

{{<katex display>}}
R(x_i)=\sum_{j}\frac{x_i p_ij}{\sum_{i'} x_{i'} p_{i'j}}R(y_j) \text{ and } R(x_i)=\sum_{j}\frac{x_i (\delta_{ij} - \frac{1}{N})}{\sum_{i'} x_{i'} (\delta_{i'j} - \frac{1}{N})}R(y_j),
{{</katex>}}


where {{<katex>}}p_{ij}{{</katex>}} is a gating term value from attention head and for the LayerNorm {{<katex>}}(\delta_{ij} - \frac{1}{N}){{</katex>}} is the other way of writing the 'centering matrix', {{<katex>}}N{{</katex>}} is the number of tokens in the input sequence.

## Better LRP Rules for Transformers

In practice authors observed that these rules do not need to be implemented explicitly. There are trick makes the method straightforward to implement, by adding `detach()` calls at the appropriate locations in the neural network code and then running standard Gradient×Input.

So improved rules will be
{{<katex display>}}
y_i = \sum_i x_i[p_{ij}].detach()
{{</katex>}}
for every attention head, and
{{<katex display>}}
y_i = \frac{x_i - \mathbb{E}[x]}{\sqrt{\varepsilon + Var[x]}}.detach()
{{</katex>}}
for every LayerNorm, where {{<katex>}} \mathbb{E}{{</katex>}} and {{<katex>}}Var[x]{{</katex>}} is mean and variance.

## Results
In the article different methods was tested on various datasets, but for now most inetersing is comparisom between old Gradient×Input (GI) method and new LRP methods with modifications in attention head rule (AH), LayerNorm (LN) or both (AH+LN).

<p align="center">
    <img src="/xai_for_transformers/2.png" width='300' alt="GIvsLRP results"/>
</p>
<h5><font color="DimGray"><center>Img.2. AU-MSE (area under the mean squared error)
</center></font></h5>

The LRP with both modifications shows slightly better results in comparison with Gradient×Input method, but may make a huge difference in the future.

The results on SST-2 dataset that contains movie reviews and ratings are shown below. Both transformers was learned to classify review as positive or negative, and LRP shows slightly brighter and more concrete relevance values.

<p align="center">
    <img src="/xai_for_transformers/4.png" alt="SST-2 results"/>
</p>

## References
[1] [Ameen Ali et. al. “XAI for Transformers: Better Explanations through Conservative Propagation.” ICML, 2022](https://proceedings.mlr.press/v162/ali22a.html)

[2] [Hila Chefer et. al. “Transformer Interpretability Beyond Attention Visualization.” CVPR, 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf)

## Code

All code for Transformer you can find in [https://github.com/AmeenAli/XAI_Transformers](https://github.com/AmeenAli/XAI_Transformers)

