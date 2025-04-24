# SHAP. Tree Explanations

## Exploring SHAP Tree Explainer for Predictive Health Analytics: A Deep Dive into RandomForest Models

Predictive models are essential for managing healthcare resources, predicting disease outbreaks, and guiding public health policy in the quickly changing field of health analytics. 
RandomForest is a standout model among these due to its solid performance, interpretability, and simplicity. But it can be difficult to comprehend how these models' complex internal workings operate. 
Presenting SHAP (SHapley Additive exPlanations), a revolutionary tool that clarifies how machine learning models—like RandomForest—make decisions. 
In-depth discussion of the SHAP Tree Explainer, its use in RandomForest models, and its importance in predictive health analytics are provided in this article.

## What is SHAP?

The unified measure of feature importance known as SHAP, or SHapley Additive exPlanations, gives each feature a value based on how important it is for a certain prediction. Basing itself on the idea of Shapley values from cooperative game theory, it provides an explanation for any machine learning model's output. After taking into consideration feature interactions, SHAP values indicate the relative contribution of each feature to the prediction for a particular instance.

## SHAP Tree Explainer for RandomForest

A customized version of the SHAP library called the SHAP Tree Explainer is intended to be used with tree-based models, such as RandomForest. Trees are more difficult to interpret since they are by nature complicated and nonlinear, in contrast to linear models. To tackle this difficulty, the SHAP Tree Explainer dissects a tree's prediction into contributions from every leaf node and attributes the prediction to the qualities that gave rise to those nodes.

## How Does It Work?

Imagine a RandomForest model predicting whether a person is infected with a disease based on various factors like age, symptoms, and medical history. The SHAP Tree Explainer starts by assigning a base value to the prediction, which is the average outcome of the model across all instances. Then, it iterates through the tree, adding the contribution of each feature along the path to the final prediction. Each feature's contribution is calculated based on its impact on the prediction, taking into account the interactions with other features.

![image](https://github.com/IU-PR/xai/assets/88908152/f30498ec-8ac4-4f66-8bec-a99a5d6ffe41)


## Benefits of Using SHAP with RandomForest

- Interpretability: SHAP provides clear, actionable insights into how each feature influences the prediction, making it easier to understand the model's decisions.
- Fairness Analysis: By showing how each feature contributes to the prediction, SHAP helps identify potential biases in the model, aiding in fairness assessments.
- Model Debugging: It highlights areas where the model might be performing poorly or where additional data could improve predictions.
- 
## How to use TreeExplainer?

1. Install the shap library
   ```python
   !pip install shap # install
   ```
2.  Read the data
   ```python
   df = pd.read_csv('/content/AIDS_Classification.csv') 
   ```
3. Initialize tree model
   ```python
   model = RandomForestClassifier(n_estimators=200, random_state=121)
   ```
4. Fit the model
   ```python
   model.fit(X_train, y_train)
   ```
5. Make a prediction
   ```python
   predictions = model.predict(X_test) 
   ```
6. Initialize TreeExplainer
   ```python
   explainer = shap.TreeExplainer(model)
   ```

7. Now, we are ready for construct graph. The first is visualise the shap values.
   ```python
   shap_values = explainer.shap_values(X) # compute common shap values and visualize
   shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type="dot", auto_size_plot=False, show=False)
   plt.show()
   ```
![image](https://github.com/IU-PR/xai/assets/88908152/6a14d69f-c442-41eb-b24a-2dc9f8b58546)


8.The second shown the SHAP interaction values between two features ("age" and "preanti") are computed, and the features' dependence on the model's predictions are displayed. This aids in comprehending the interplay between these two characteristics while forecasting.

   ```python
   shap_interaction = explainer.shap_interaction_values(X) # compute interaction values and visualize
   shap.dependence_plot(("age", "preanti"), new_shap_interaction, X, feature_names=X.columns, show=False)
   plt.show()
   ```

This graph shows that those who use drugs and are over 40 have a higher risk of contracting an illness.

![image](https://github.com/IU-PR/xai/assets/88908152/f835a7f9-902d-437f-a83f-683711f8ab1b)



## Conclusion

Our understanding of and confidence in predictive health analytics models is greatly improved by integrating SHAP with RandomForest models. An essential tool for healthcare decision-making, **SHAP's Tree Explainer provides comprehensive insights into how specific attributes affect forecasts**. In addition to its interpretability, SHAP facilitates fairness analysis and debugging of models, which improves the model development process. The essay showcases SHAP's transformative potential in turning raw data into actionable knowledge through its practical application, which primarily focuses on forecasting disease infection status. In order to achieve transparency and reliability in predictive models going forward, the field will need to adopt SHAP, which will be crucial.


Link to our colab:

[Code example](https://colab.research.google.com/drive/1hfdtyhN8ENk49Y-zTB2IH06VQmbeBenl?usp=sharing)

Link to the dataset:

[AIDS_classification](https://www.kaggle.com/datasets/aadarshvelu/aids-virus-infection-prediction?resource=download)

Links:

[Documentation of SHAP](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html#)

[Paper (Explainable AI for Trees: From Local Explanations to Global Understanding, Scott M. Lundberg et al.)](https://paperswithcode.com/paper/explainable-ai-for-trees-from-local)

[Official exaple of usage](https://github.com/suinleelab/treeexplainer-study/blob/master/README.md)

[SHAP on github](https://github.com/shap/shap)


