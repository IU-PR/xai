---
weight: 1
bookFlatSection: true
title: "Dimensionality Reduction in NLP: Visualizing Sentence Embeddings with UMAP and t-SNE"
---

# Dimensionality Reduction in NLP: Visualizing Sentence Embeddings with UMAP and t-SNE

## Students:
- Yazan Kbaili (y.kbaili@innopolis.university)
- Hamada Salhab (h.salhab@innopolis.university)

## Introduction
This report delves into the visualization of sentence embeddings derived from the `roberta-base` model fine-tuned on the "go-emotion" dataset using two prominent dimensionality reduction techniques: UMAP (Uniform Manifold Approximation and Projection) and t-SNE (t-distributed Stochastic Neighbor Embedding). The dataset used in the visualization consists of Twitter messages and is called “emotions”.

## Methodology
The embeddings were generated by the `roberta-base` model, specifically tailored for multiclass emotion classification and accessible via Hugging Face. This model is particularly adept at interpreting emotional contexts within text, making it highly suitable for our study. We focus on embeddings from Twitter messages categorized into six distinct emotions.

### UMAP
Operates on a foundation of algebraic topology, creating a high-dimensional graph representation of data before optimizing this layout in a lower-dimensional space. It aims to preserve both local and global structures, offering a comprehensive data understanding. UMAP is generally preferred for its ability to maintain a more global structure compared to t-SNE.

### t-SNE
Transforms high-dimensional Euclidean distances between points into conditional probabilities that reflect similarities, excelling in the preservation of local data structures and the identification of clusters. While effective, t-SNE can be computationally demanding, especially with large datasets, and may exaggerate cluster separations without maintaining global data integrity.

### UMAP vs t-SNE

The behavior of t-SNE and UMAP differs significantly in terms of initialization and the iterative process used to optimize the low-dimensional representation. Below are some of the key differences:

| Feature             | t-SNE                                               | UMAP                                                              |
|---------------------|-----------------------------------------------------|-------------------------------------------------------------------|
| **Initialization**  | Starts with a random initialization of the graph.   | Uses Spectral Embedding for deterministic initialization.         |
| **Iteration Process** | Moves every single point slightly each iteration.  | Can move just one point or a small subset of points each time.    |
| **Scalability**     | Less efficient with very large datasets.            | Scales well with large datasets due to its partial update approach. |

## Extraction of Embeddings
We utilized the embedding from the last token at the last layer of the pre-trained model, which typically encapsulates the sentence's contextual essence.

## Visualization
For an interactive visual representation, we employed Plotly, enabling detailed exploration of the structures within the embeddings.


### t-SNE
1:
![image](https://hackmd.io/_uploads/Hy0VHwm7R.png)
2:
![t-SNE](https://hackmd.io/_uploads/H1a6ul7mC.png)
### UMAP
1:
![UMAP1](https://hackmd.io/_uploads/BkaTOg77A.png)
2:
![UMAP2](https://hackmd.io/_uploads/rJ66_xm7C.png)

## Insights

The visualizations above show that the dimensionality reduction methods used did a reasonably good job with clustering the different labels of data. Note that the model is trained on a different dataset, and has never seen the data we tried it, which explain the limiatation and 

## Extra Insights

Let's take a look at this 2D visualization from UMAP:

![telegram-cloud-photo-size-4-5922267642553550056-y](https://hackmd.io/_uploads/ryWD9P7QR.jpg)

- We zoomed in to the cluster highlighted in red, and found out that altough it contains sentences that belong to all labels, all of them convey some sort of regret/guilt.

![image](https://hackmd.io/_uploads/Hk25Xv7XC.png)

- We took another look at the cluster highlighted in yellow, and saw that all the sentences there talk about humiliation/disgrace.
![image](https://hackmd.io/_uploads/r1pTwD7XC.png)

- As for the cluster highlighted in black, all the sentences conveyed meaning for the purpose of thanking and appreciation.
![image](https://hackmd.io/_uploads/Hyw39PXQ0.png)



## Applications
- **Model Debugging and Improvement**: Identifies anomalies or biases in embeddings, facilitating targeted improvements to the model.
- **Semantic Analysis**: Assists in understanding how sentences are clustered semantically, which helps in tasks such as sentiment analysis.
- **Transfer Learning Insights**: Offers insights into the transferability of learned features, especially in domain-specific applications.
- **Multilingual Comparisons**: Evaluates the model’s capability across different languages, which helps identify potential biases or gaps.
- **Explainability and Trust**: Increases the transparency of NLP systems, and builds trust among end-users and regulators by making complex models more interpretable.

## Colab Notebook
The source code for this project can be found in this [Colab Notebook](https://colab.research.google.com/drive/1rs08XMn38GFz2bcOKXdCJJUKrh8rgNHF#scrollTo=cSbyb4wbtCZn).

## Presentation Slides

The presentation slides for this project can be found on [Google Slides](https://docs.google.com/presentation/d/1Bclphb2ixuRuoXJU4qDmzm6ohTTB1fOEBakchy19y4g/edit?usp=sharing).