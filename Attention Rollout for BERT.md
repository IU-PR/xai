# BERT Sentiment Prediction Explanation with Attention Rollout

## Introduction

BERT (Bidirectional Encoder Representations from Transformers) model is widely used for text analysis and serves as an example of popular and important pre-trained models. It also perfectly illustrates attention mechanism, which in turn serves as a key element of an important stage of ML development. 

Nethertheless, as most of the complex ML models, BERT is uninterpretable for human mind and perspective. It damages both effectiveness and trust in work within ML field and thus is addressed by XAI. 

As I find sentiment analysis task both interesting in terms of potential insights into the nature of human language and important in terms of services based around it, such as advertisements industry, I decided to find an explanation of BERT sentiment prediction with Attention Rollout XAI method.

## Initial Hypothesis
It seems reasonable to assume that the most important and the most attended tokens during sentiment analysis tasks are tokens of words bearing strong evaluational or judgemental weight generally or in the given context. In this case, it is expected that words connected to art analysis (e.g. deep, meaningful etc) and entertainment (fun, boring) would dominate the attention field.

## Bidirectional Encoder Representations from Transformers (BERT)

### Conceptual Foundation
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model that learns contextualized word representations through self-supervised pre-training. For NLP tasks, it provides:

- Contextual understanding through bidirectional attention mechanisms

- Transfer learning capabilities via pre-training on large corpora

- State-of-the-art performance across diverse NLP tasks

### Training
As model used is pre-trained on large corpora of English text already, its further re-training on the specific dataset was mostly performed for experiment, which showed low correlation of train and test accuracy and is not important for this project.

### Performance
Model shows sufficiently high accuracy when predicting sentiment of specific reviews, approaching 90% of correct predictions.

## Attention Rollout Analysis
### Conceptual Foundation:
Attention rollout visualizes how information flows through transformer layers by:
- Aggregating attention weights across all heads
- Tracking attention propagation through layers
- Identifying influential input tokens for predictions

### Why Attention Rollout?
Attention rollout was chosen as an XAI method for this project due to its focus on attention mechanism which I find to be especially important and interesting. 

As opposed to, e.g, attention flow method, attention rollout was chosen due to its sufficiently low computation time which was welcome due to low amount of computational resources and time available. 

## Data
### IMDB dataset
IMDB movie review dataset contains texts and labels (positive or negative) of movie reviews. It is aimed for sentiment analysis and is chosen for this project because it is both simple and representative of modern conversational English language which poses special interest for research and thus is a fitting object of analysis.

## Results
Attention rollout shows that owerhelming majority of attention is given to the punctuation marks which specify the end of sentence, specifically dots. They consistently dominate the tops of the most influential tokens in all the performed tests on randomly chosen subsets of IMDB database.

Excluding random outliers, otherwise BERT considers important primely adjectives bearing strong judgemental (positive or negative) weight or associated with movie quality (boring, fun etc.) which still show much less weights than punctuation marks.

## Conclusion
Implementation of attention rollout from scratch proved to be interesting and helpful in terms of learning internal workings of ML models and proved itself to be another tool of furthering AI explainability. 

Results of the project show that despite initial hypothesis, BERT focuses primarily not on the tokens directly meaningful for the task of sentiment analysis, but on tokens which provide understanding of text's internal structure.
<!--more-->
 


## References
[1] [Attention Rollout original article](https://arxiv.org/pdf/2005.00928)

[2] [GitHub page](https://github.com/AlSovPich/XAI-project)
