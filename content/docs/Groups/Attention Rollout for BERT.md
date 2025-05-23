# BERT Sentiment Prediction Explanation with Attention Rollout

## Introduction

BERT (Bidirectional Encoder Representations from Transformers) model is widely used for text analysis and serves as an example of popular and important pre-trained models. It also perfectly illustrates attention mechanism, which in turn serves as a key element of an important stage of ML development. 

Nethertheless, as most of the complex ML models, BERT is uninterpretable for human mind and perspective. It damages both effectiveness and trust in work within ML field and thus is addressed by XAI. 

As I find sentiment analysis task both interesting in terms of potential insights into the nature of human language and important in terms of services based around it, such as advertisements industry, I decided to find an explanation of BERT sentiment prediction with Attention Rollout XAI method.

## Initial Hypothesis
It seemed reasonable to assume that the most important and the most attended tokens during sentiment analysis tasks are tokens of words bearing strong evaluational or judgemental weight generally or in the given context. In this case, it is expected that words connected to art analysis (e.g. deep, meaningful etc) and entertainment (fun, boring) would dominate the attention field.

## Bidirectional Encoder Representations from Transformers (BERT)
[![BERT](https://github.com/AlSovPich/xai_attention_rollout_for_bert/blob/patch-1/content/docs/Groups/images/bert.png "BERT")](https://github.com/AlSovPich/xai_attention_rollout_for_bert/blob/patch-1/content/docs/Groups/images/bert.png "BERT")

### Conceptual Foundation
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model that learns contextualized word representations through self-supervised pre-training. For NLP tasks, it provides:

- Contextual understanding through bidirectional attention mechanisms

- Transfer learning capabilities via pre-training on large corpora

- State-of-the-art performance across diverse NLP tasks

### Core Components
BERT is a stack of transformer encoder blocks, each consisting of:

- Multi-head Self-Attention
- Layer Normalization
- Feedforward Neural Network (FFN)
- Residual Connections

The attention mechanism at the heart of BERT is given by:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- \( Q, K, V \) are query, key, and value matrices
- \( d_k \) is the dimension of the key vector

Multi-head attention splits these matrices into subspaces to learn diverse relationships:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$


---

### Training
Despite used model being pre-trained on large corpora of English texts, it was important for the sake of the experiment to re-train it on the used dataset. Thus, re-training was performed to allow for analysis of how it would assess and attend tokens in this task specifically, acknowledging for actual patterns of modern conversational language used in movie reviews and similar environments.

### Performance
Model shows sufficiently high accuracy when predicting sentiment of specific reviews, approaching 90% of correct predictions.

### Specifics in terms of Attention
BERT uses special [CLS] (classification) and [SEP] (separation) tokens at the beginning and at the end of each sentence respectively for the tasks requiring analysis not of individual words, but of sentences and whole texts. [CLS] token starts as meaningless, but later absorbs information about text segments it opens and in the end serves as an embedding of whole segment's meaning. As this project deals with sentiment analysis, [CLS] token stands to be especially important. If accounted for, [CLS] completely dominates attention field, which gives no information on the actual BERT workings and attentions. Thus, service tokens standing for intra-model concepts and not text entities were removed from consideration of attention weights.

## Attention Rollout Analysis
### Conceptual Foundation:
Attention rollout visualizes how information flows through transformer layers by:
- Aggregating attention weights across all heads
- Tracking attention propagation through layers
- Identifying influential input tokens for predictions

### Mathematical foundation
[![Formula](https://github.com/AlSovPich/xai_attention_rollout_for_bert/blob/patch-1/content/docs/Groups/images/formula.png "Formula")](https://github.com/AlSovPich/xai_attention_rollout_for_bert/blob/patch-1/content/docs/Groups/images/formula.png "Formula")


### Why Attention Rollout?
Attention rollout was chosen as an XAI method for this project due to its focus on attention mechanism which I find to be especially important and interesting. Attention analysis allows to detect specific patterns model is looking for in the provided data, thus providing a great direction for research in the explainability field. Explainability is based on bridging the gap between machine and human understanding of data, and understanding how machine attends - breaks down and prioritizes - data is important for it.

As opposed to, e.g, attention flow method, attention rollout was chosen due to its sufficiently low computation time which was welcome due to low amount of computational resources and time available. As this project is aimed not on achieving particular practical tasks, but at researching existing method and model in tandem, understanding basics of XAI and AI overall, using computationally heavy methods would be highly inconvenient, ineffective and unresultful.

### Implementation
```python
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
cumulative_attention = torch.eye(len(tokens))

for layer_attention in attentions:
	avg_attention = layer_attention.mean(dim=1).squeeze(0)
	avg_attention += torch.eye(avg_attention.size(0))
	avg_attention /= avg_attention.sum(dim=-1, keepdim=True)
	cumulative_attention = torch.matmul(avg_attention, cumulative_attention)

cumulative_attention = cumulative_attention.detach().numpy()

token_to_indices = defaultdict(list)
for idx, token in enumerate(tokens):
	if token not in ['[PAD]', '[SEP]', '[CLS]']:
		token_to_indices[token].append(idx)

unique_tokens = list(token_to_indices.keys())
aggregated_attention = np.zeros((len(unique_tokens), len(unique_tokens)))

for i, token_i in enumerate(unique_tokens):
	indices_i = token_to_indices[token_i]
	for j, token_j in enumerate(unique_tokens):
		indices_j = token_to_indices[token_j]
		aggregated_attention[i, j] = cumulative_attention[np.ix_(indices_i, indices_j)].mean()

aggregated_attention = aggregated_attention / aggregated_attention.sum(axis=1, keepdims=True)
```

## Data
### IMDB dataset
IMDB movie review dataset contains texts and labels (positive or negative) of movie reviews. It is aimed for sentiment analysis and is chosen for this project because it is both simple and representative of modern conversational English language which poses special interest for research and thus is a fitting object of analysis.

## Results
Results of the project show that BERT highly prioritizes two kinds of tokens: those important for structural analysis of text but not bearing emotional stance themselves (initiating), and those which have strong judgemental meaning (evaluational). These initiating words consistently attract higher attention, indicating BERT's sensitivity to phrase-level sentiment structure. It partially disproves initial hypothesis, showing that BERT primarily assesses not individual words, but meaningful sequences. Still, evaluational tokens also bear quite a significant weight.

### Visualization
```python
top_words, top_scores = aggregate_attention_scores(explanations, top_k)
    
# Calculate accuracy statistics
correct = sum(1 for exp in explanations if exp['correct']) / len(explanations)
pos_probs = [exp['pos_prob'] for exp in explanations]
neg_probs = [exp['neg_prob'] for exp in explanations]
avg_pos_prob = sum(pos_probs) / len(pos_probs)
avg_neg_prob = sum(neg_probs) / len(neg_probs)

# Create horizontal bar plot
plt.figure(figsize=(12, 8))
bars = plt.barh(range(top_k), top_scores, align='center', color='skyblue')
plt.yticks(range(top_k), top_words)
plt.xlabel('Average Normalized Attention Score')
plt.gca().invert_yaxis()
plt.title(
	f"Top {top_k} Important Words Across All Samples\n"
	f"Average Positive Probability: {avg_pos_prob:.2f}, "
	f"Average Negative Probability: {avg_neg_prob:.2f}\n"
	f"Overall Accuracy: {correct:.2%}"
)

# Add value labels to bars
for bar in bars:
	width = bar.get_width()
	plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
	ha='left', va='center')

plt.tight_layout()
plt.show()
```

[![Graph](https://github.com/AlSovPich/xai_attention_rollout_for_bert/blob/patch-1/content/docs/Groups/images/output.png "Graph")](https://github.com/AlSovPich/xai_attention_rollout_for_bert/blob/patch-1/content/docs/Groups/images/output.png "Graph")

## Conclusion
Implementation of attention rollout from scratch proved to be interesting and helpful in terms of learning internal workings of ML models and proved itself to be another tool of furthering AI explainability. 

Results of the project show that despite initial hypothesis, BERT focuses primarily not on the tokens directly meaningful for the task of sentiment analysis, but on tokens which provide understanding of text's internal structure. Emotional phrases often begin with highly attended tokens, revealing BERT's compositional understanding and awareness as well as a tendency in human language towards more complex linguistical structures.


## References

[1] Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (https://arxiv.org/pdf/2005.00928)

[2] Project GitHub Repository: https://github.com/AlSovPich/XAI-project
