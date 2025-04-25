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
As model used is pre-trained on large corpora of English text already, its further re-training on the specific dataset was mostly performed for experiment, which showed low correlation of train and test accuracy and is not important for this project.

### Performance
Model shows sufficiently high accuracy when predicting sentiment of specific reviews, approaching 90% of correct predictions.

## Attention Rollout Analysis
### Conceptual Foundation:
Attention rollout visualizes how information flows through transformer layers by:
- Aggregating attention weights across all heads
- Tracking attention propagation through layers
- Identifying influential input tokens for predictions

### Mathematical foundation
Given attention matrices $$\( A^1, A^2, ..., A^L \) for \( L \)$$ layers, the rollout is computed as:

$$\[
\tilde{A}^l = \frac{1}{H} \sum_{h=1}^H A_h^l + I \\
\tilde{A}^l = \frac{\tilde{A}^l}{\sum_j \tilde{A}^l_{ij}} \text{ (row norm)} \\
R = \tilde{A}^L \cdot \tilde{A}^{L-1} \cdot ... \cdot \tilde{A}^1
\]$$

Where $$\( H \)$$ is the number of heads. This gives us a final attention matrix from each input token to the $$[CLS]$$ token.

### Why Attention Rollout?
Attention rollout was chosen as an XAI method for this project due to its focus on attention mechanism which I find to be especially important and interesting. 

As opposed to, e.g, attention flow method, attention rollout was chosen due to its sufficiently low computation time which was welcome due to low amount of computational resources and time available. 

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
Upon deeper inspection, emotionally and judgmentally charged words (such as great, boring, terrible) often appear at the beginning of key evaluative phrases (e.g., "great movie", "terribly acted"). These initiating words consistently attract higher attention, indicating BERT's sensitivity to phrase-level sentiment structure. It partially disproves initial hypothesis, showing that BERT primarily assesses not individual words, but meaningful sequences.

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

## Conclusion
Implementation of attention rollout from scratch proved to be interesting and helpful in terms of learning internal workings of ML models and proved itself to be another tool of furthering AI explainability. 

Results of the project show that despite initial hypothesis, BERT focuses primarily not on the tokens directly meaningful for the task of sentiment analysis, but on tokens which provide understanding of text's internal structure. Emotional phrases often begin with highly attended tokens, revealing BERT's compositional understanding and awareness as well as a tendency in human language towards more complex linguistical structures.


## References

[1] Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (https://arxiv.org/pdf/2005.00928)

[2] Project GitHub Repository: https://github.com/AlSovPich/XAI-project
