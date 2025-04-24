# Unveiling black boxes with SHAP Values
Nowadays, correct interpretation of model predictions is crucial. It builds user confidence, helps to understand the process being modeled, and suggests how to improve the model. Sometimes simple models are preferred, e.g. in finance, because they are easy to interpret, but they usually do not achieve the same performance as complex ones. 
Therefore, to overcome this trade-off between accuracy and interpretability, various methods have been developed. In this post I want to talk about one of them, the SHAP framework.
# What are SHAP Values?

SHAP Values is a method that assigns each feature a value that reflects its contribution to the model prediction. These values are based on cooperative game theory, [the concept of Shapley values](https://en.wikipedia.org/wiki/Shapley_value), introduced by Lloyd Shapley.
In this context, each attribute is treated as a player in the game, and the Shapley value measures the average marginal contribution of each attribute across all possible combinations of attributes
# The Mathematics Behind SHAP Values

As already mentioned, Shapley values is a concept in the theory of cooperative games. For each such game, it specifies the distribution of the total payoff received by the coalition of all players.
### Formal Definition:
The Shapley value for player {{<katex>}} i {{</katex>}} in a cooperative game is defined as the average marginal contribution of the player to the coalitions.
Formally, we have a set {{<katex>}}N{{</katex>}} of players and a characteristic function {{<katex>}}\mathcal{v}{{</katex>}}  representing gains, which maps subset of players to real numbers(gain). Also, {{<katex>}}\mathcal{v(\emptyset)}=0{{</katex>}} meaning that empty coaliation of players worths nothing.
Then, the Shapley value {{<katex>}}\phi_i{{</katex>}} for player {{<katex>}}i{{</katex>}} is given by:
{{<katex display>}}
\phi_i{(\mathcal{v})}=\sum_{S \subset N \backslash \left\{i\right\}}{\frac{|S|!(n-|S|-1)!}{|N|!} (\mathcal{v}(S \cup \left\{i\right\})-\mathcal{v}(S))}
{{</katex>}}

where:
* {{<katex>}}S{{</katex>}} is a subset of players excluding player {{<katex>}}i{{</katex>}}.
* {{<katex>}}|S|{{</katex>}} is the number of player in the coalition {{<katex>}}S{{</katex>}}
* {{<katex>}}\mathcal{v}(S){{</katex>}} is a total gain of the coalition {{<katex>}}S{{</katex>}}
This formula calculates the marginal contribution of player {{<katex>}}i{{</katex>}} to each possible coalition and then averages it.
### Extension to SHAP Values and Properties:

SHAP combines the local interpretability methods(Linear LIME, for example) and Shapley values. It results in desired properties:
1. Local accuracy: {{<katex display>}}f(x)=g(x')=\phi_0+\sum^M_{i=1}{\phi_ix'_i}{{</katex>}} The explanation model {{<katex>}}g(x'){{</katex>}} matches the original model {{<katex>}}f(x){{</katex>}} when {{<katex>}}x=h_x(x'){{</katex>}}, where {{<katex>}}\phi_0=f(h_x(0)){{</katex>}} represents the model output with all simplified inputs toggled off(missing)
2. Missingness:{{<katex display>}}x'_i=0 \to \phi_i=0{{</katex>}} Missing features have no **attributed** impact
3. Consistency: Let {{<katex>}}f_x(z')=f(h_x(z')){{</katex>}} and {{<katex>}}z'\backslash i{{</katex>}} denote setting {{<katex>}}z'_i=0{{</katex>}}. For any two models {{<katex>}}f{{</katex>}} and {{<katex>}}f'{{</katex>}}, if {{<katex display>}}f'_x(z')-f'_x(z'\backslash i)\geq f_x(z')-f_x(z'\backslash i){{</katex>}} for all inputs {{<katex>}}z' \in \left\{0,1\right\}^M{{</katex>}}, then {{<katex>}}\phi_i(f',x)\geq\phi_i(f,x){{</katex>}}.
It means that if a model changes so that the marginal contribution of a feature value increases or stays the same (regardless of other features), the Shapley value also increases or stays the same.
# Computation of SHAP Values
### Kernel SHAP
This is a model-agnostic method for approximating SHAP values. This method uses a Linear LIME to locally approximate the original model.

First, we need to heuristically choose the parameters for LIME:
{{<katex display>}}\Omega(g)=0,{{</katex>}}
{{<katex display>}}\pi_{x}(z')=\frac{(M-1)}{(M \text{ choose } |z'|)|z'|(M-|z'|)}{{</katex>}}
{{<katex display>}}L(f,g,\pi_{x})=\sum_{z'\in Z}{[f(h_x(z'))-g(z')]^2\pi_{x}(z')}{{</katex>}}

Then, since {{<katex>}}g(z'){{</katex>}} is linear, {{<katex>}}L{{</katex>}} is a squared loss, the objective function of LIME: {{<katex>}}\xi=\underset{g \in \mathcal{G}}{\operatorname{argmin}}{L(f,g,\pi_{x'})+\Omega{(g)}}{{</katex>}} can be solved using linear regression.
### Illustrative example

1. Model and Instance:
Let's say we have a predictive model f and a dataset with three features. We want to understand how each feature contributes to the model's prediction for a specific data point {{<katex>}}x = (x_1,x_2,x_3){{</katex>}} by computing SHAP values.
2. Generating coalitions:
To do it we need to consider all possible coalitions of features that could be used to make a prediction. Each coalition is a subset of the features used for prediction. The set of coalitions in our case: {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}.
3. Obtaining modeling results for coalitions:
For each of these coalitions, we compute the model output. The missing features must be imputed. We obtain the following outputs: {{<katex>}}f(\emptyset), f(x_1),f(x_2),f(x_3),f(x_1,x_2),f(x_1,x_3),f(x_2,x_3),f(x_1,x_2,x_3){{</katex>}}
4. Obtaining weights for coalitions:
{{<katex display>}}\pi_{x}(z')=\frac{(M-1)}{(M \text{ choose } |z'|)|z'|(M-|z'|)}{{</katex>}}
For example, {{<katex>}}\pi_x({0,0,1})=\frac{3-1}{\frac{3!}{1!(3-1)!}1(3-1)}=\frac{2}{6}=\frac{1}{3}{{</katex>}} 
5. We obtain the model {{<katex>}}g{{</katex>}}:
Finally, we train a linear model (explanation model) {{<katex>}}g{{</katex>}}. This model is trained to match the outputs of our original model {{<katex>}}f{{</katex>}}. The weights of the model {{<katex>}}g{{</katex>}} are obtained by optimizing the following loss function {{<katex display>}}L(f,g,\pi_{x})=\sum_{z'\in Z}{[f(h_x(z'))-g(z')]^2\pi_{x}(z')}{{</katex>}} The weights of model {{<katex>}}g{{</katex>}} are the Shapley values.
# Interpreting SHAP Values
### Individual Instance Interpretation:
* **Feature Contribution:** SHAP values give us the ability to measure how badly or good a feature is in making model predictions about an individual, instance. A positive value implies that the feature contributes towards increasing the prediction of the model while a negative value suggests that it reduces the prediction.
* **Magnitude**: The magnitude or absolute value of a SHAP number impacts on how much influence a particular attribute has in our model prediction, Larger numbers indicate more importance in shaping up the final outcome.
### Global Feature Importance:

* **Feature Importance Ranking:** The average of the absolute SHAP values for each feature across all instances gives us a global ranking of feature importance. This ranking helps identify the features that consistently have the most significant impact on model predictions.
* **Understanding Model Behavior:** An insight into how well our model reacts to different levels of a given characteristic is only possible by studying distributions of its corresponding Shapley Values. In doing so, an expose can be made about any prejudices or non-linear aspects within.
# Applications of SHAP Values
SHAP values is a powerful tool that has several applications:

**1. Model Debugging:** we can identify features which cause problems in predictions, that can indicate, for example, data leakage or correlations.

**2. Fairness and Bias Analysis:** we can identify biases when the model making different unfair predictions based on some attributes like race, gender, and etc. Understanding the impact of these feature can help us develop fairer models.
# Conclusion:

### Main takeaways:
* **Shapley values is a measure of the average marginal contribution of each feature across all possible subsets of features.**
* **SHAP values are connecting Shapley values to the local interpretability methods, providing properties such as locality, missingness, and consistency.**
* **SHAP values enable us to better undestand both individual instance and global model behavior, as well as providing feature contribution analysis, feature importance ranking, and model behavior understanding.**
* **They are commonly used for model debugging allowing identifying problematic features and developing more unbiased and fair model**
# References:
The blog post is mainly based on the original paper introducing SHAP values - [A Unified Approach to Interpreting Model Predictions (nips.cc)](https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

The original implementation of SHAP - [shap/shap: A game theoretic approach to explain the output of any machine learning model. (github.com)](https://github.com/shap/shap)

My implementation of Kernel SHAP - https://colab.research.google.com/drive/1TPHvns2psDNKknwubxCTHZprw3UGB-w1?usp=sharing
