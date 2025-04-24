# Integrated Gradients Method for Image Classification

**XAI Course Project | Anatoliy Pushkarev**

## Goal

Develop a robust image classification model and analyze its behavior with the help of the integrated gradients method.

[Integrated gradients paper](https://arxiv.org/abs/1703.01365)

## Integrated Gradients

Integrated Gradients is a technique for attributing a classification model's prediction to its input features. It is a model interpretability technique: you can use it to visualize the relationship between input features and model predictions. It finds the importance of each pixel or feature in input data for a particular prediction of the model.

Integrated gradients original paper can be found [here](https://arxiv.org/abs/1703.01365).

### Steps

1. Select baseline (uniform distribution for all classes).
2. Evaluate the path from baseline to input data point by many iterations.
3. Observe how changing input data affects gradients.
4. Integrate all gradients.

This is the original integrated gradients formula:
![Original Integrated Gradients formula](https://www.dropbox.com/scl/fi/iheoqe68yvb4rgk99rqd3/2024-05-24-09.39.57.png?rlkey=78wwqc9xwk5zt7l101kejxzj3&st=vf2foy7b&dl=1)


This is the Riemann approximation of the original formula, which is always used.
![Approximation of the Original Integrated Gradients formula](https://www.dropbox.com/scl/fi/iheoqe68yvb4rgk99rqd3/2024-05-24-09.39.57.png?rlkey=78wwqc9xwk5zt7l101kejxzj3&st=vf2foy7b&dl=1)

A very good article about integrated gradients and Riemann Approximation can be found here:
- [Great IG method explanation](https://distill.pub/2020/attribution-baselines/)

This is a good picture which shows, why exactly we need to sum gradients. Basically if we just take the gradients of the model wrt the inputs, we will get a lot of noise, which is illustrated on the right side of the picture. But if we scale the inputs, at the some point we get interesting gradients, which the method uses.

![How IG method works, why scale](https://www.dropbox.com/scl/fi/wnkhibb5476z0g6bwg64h/2024-05-24-09.40.41.png?rlkey=ma6pk54gtgzpoiso6980nhfma&st=kj4ldpuu&dl=1)


## MNIST Fashion Dataset

The dataset which I am using for the test purposes is Fashion-MNIST. Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

![Fashion-MNIST dataset](https://www.dropbox.com/scl/fi/qggb41s30tq8829uul714/2024-05-24-09.40.31.png?rlkey=2as21m78o53kw5ofyrqn3c55l&st=24vyd52t&dl=1)


## Model

- Keras Sequential
- 3 Conv layers
- 2 Max pooling layers
- Dropouts
- 2 FC layers

![Model architecture](https://www.dropbox.com/scl/fi/l2avsyvx284nwywxhk1s7/2024-05-24-09.40.59.png?rlkey=edolnwdfjmmvfi9sickakvict&st=oyk9q2x6&dl=1)

## Training

Below you can find a screenshot of model training loop. It is pretty straightforward.

![Model training loop](https://www.dropbox.com/scl/fi/c4j3ifjeqv9wla7lznxdx/2024-05-24-09.41.06.png?rlkey=2e8ztnbwne6qlhrv05a5006qr&st=xjsppae0&dl=1)

## Results of Training

As a result, I got a robust model with accuracy > 90%, which is good for explanation.

![Results of training](https://www.dropbox.com/scl/fi/7ea8iwotib8junpad2lsw/2024-05-24-09.41.28.png?rlkey=ienzj5e7khphc6wvzrjrv83c6&st=go9ppgo5&dl=1)

There are some classes with lower precision and recall, it will be interesting to check IGs of them (for example, Shirt classs).

Examples of correctly predicted classes:
![Correctly predicted classes](https://www.dropbox.com/scl/fi/ajf8itfsaal7xgq88hkf5/2024-05-24-09.41.37.png?rlkey=fhnh2pd2wdtd9fvwguzq7bv52&st=t9q53mwl&dl=1)

Examples of incorrectly predicted classes:
![Incorrectly predicted classes](https://www.dropbox.com/scl/fi/w955alns3l48o1h4rut76/2024-05-24-09.41.41.png?rlkey=3vpw3qqvspf19etkt9pm23lpa&st=c2l30vrq&dl=1)


## Applying Integrated Gradients

Below you can find maps for IGs, which I`ve got after the method execution. Basically, red means that these points are important for certain class and blue means that these are negative features for a particular class.

![IG examples](https://www.dropbox.com/scl/fi/wc8nue1m329g52wr4bevq/2024-05-24-09.41.51.png?rlkey=s1gvne2bhti1wqwph7p8vytyt&st=03xiuqcy&dl=1)

![IG examples](https://www.dropbox.com/scl/fi/f8ean846q9796nhhn3vsb/2024-05-24-09.41.58.png?rlkey=e7rxs252ieqceivfkvot2tn57&st=oyjldqbh&dl=1)

## More Examples

This is and example of how IGs work with regression tasks. Basically x-axis is features and y-axis is weights of the feature to the output.

![IG examples](https://www.dropbox.com/scl/fi/mumi06hake9n64say48iv/2024-05-24-09.42.05.png?rlkey=z4fjb6hkm1nacmwvp3qgjgb4l&st=dhk31a8a&dl=1)

## Challenges

- Lack of open-source stable solutions for any model.
- Most likely you have to implement your own solution for a complicated model.
- Lack of support and community – not so popular (SHAP and GradCAM paper cited 20K times, IG – 5K times).
