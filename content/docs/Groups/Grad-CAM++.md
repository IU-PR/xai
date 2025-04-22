---
weight: 1
bookFlatSection: true
title: "Grad-CAM++"

---
<!-- <style> .markdown a{text-decoration: underline !important;} </style>
<style> .markdown h2{font-weight: bold;} </style> -->

# Implementation of Grad-CAM++ and Prediction Explanation for a ResNet Model in Object Detection and Classification

**Authors: Elena Tesmeeva, Nazgul Salikhova**

## Explainable AI (XAI): An Overview

As machine learning models grow more complex, it becomes harder to understand *why* they make certain predictions. This black-box nature isn’t always a problem in low-risk settings, but in critical fields like healthcare or finance, lack of transparency can be risky or even harmful.

**Explainable AI (XAI)** is a set of techniques designed to shed light on a model’s decision-making process. The idea is to go beyond just high accuracy—to also understand *what* parts of the input the model focused on, *how* it reached a conclusion, and *whether* the reasoning seems reasonable or flawed.

XAI methods help build trust in models, allow for better debugging, and support regulatory requirements (like GDPR). Some popular tools and methods in this area include:

- **Saliency Maps** – Highlight which pixels or input features contributed most to a prediction.
- **LIME** (Local Interpretable Model-agnostic Explanations) – Explains individual predictions by approximating the model locally with an interpretable one.
- **SHAP** (SHapley Additive exPlanations) – Assigns importance scores to features based on cooperative game theory.
- **Grad-CAM** and **Grad-CAM++** – Visualize which regions of an image a CNN focused on when making a classification.

In essence, XAI helps open the *black box* of deep learning and makes models not just powerful, but also understandable and trustworthy.

---


## Overview of Grad-CAM++

Grad-CAM++ (Gradient-weighted Class Activation Mapping++) is an advanced version of Grad-CAM, developed to provide improved visual explanations for decisions made by convolutional neural networks (CNNs). Grad-CAM works by utilizing the gradients of a target concept flowing into the final convolutional layer to produce a hard localization map highlighting important regions in the image.

Grad-CAM++, however, refines this approach by using second and third-order derivatives of the output with respect to the convolutional feature maps. This makes Grad-CAM++ better suited for:

- Images containing multiple object instances
- Improving localization of class-specific regions
- Generating finer, more precise heatmaps

#### Key Equation:
The key equation for Grad-CAM++ is the following:

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<div align="center">
  $$
  \text{Grad-CAM}^{++}(\mathbf{A}, \mathbf{C}) = \sum_{i} \sum_{j} \alpha_{ij} \cdot \text{ReLU}\left(\sum_{k} \frac{\partial y_{C}}{\partial A_{ijk}} \right) \cdot A_{ijk}
  $$
</div>

**Where:**  
- **A** is the feature map from the last convolutional layer.  
- **C** is the target class.  
- **α<sub>ij</sub>** represents the importance of each activation after applying the Taylor expansion.  
- **A<sub>ijk</sub>** is the activation value for each location in the feature map.  
- The gradients **∂y<sub>C</sub>/∂A<sub>ijk</sub>** are used to compute the weight of each feature map for class C.

The main idea behind Grad-CAM++ is to assign better importance weights to the feature maps in the last convolutional layer, leading to more accurate explanations without retraining or modifying the model.

---

## Comparison: Grad-CAM vs Grad-CAM++ vs CAM

| Feature                  | Grad-CAM         | Grad-CAM++       | CAM        |
|--------------------------|------------------|------------------|------------------|
| Uses gradients           | Yes              | Yes              | No               |
| Higher-order derivatives | No               | Yes              | No               |
| Forward pass only        | No               | No               | Yes              |
| Localization accuracy    | Medium           | High             | High             |
| Handles multiple objects | Poor             | Good             | Good             |
| Computational efficiency | High             | Moderate         | Low              |

Grad-CAM++ builds on the strengths of Grad-CAM by using more complex derivatives to better weigh the importance of each activation. While Score-CAM avoids gradients altogether and uses activation maps weighted by class scores, it is more computationally intensive due to multiple forward passes. Grad-CAM++ thus strikes a balance between interpretability and efficiency.

![An-overview-of-all-the-three-methods-CAM-Grad-CAM-GradCAM-with-their-respective](/Grad-CAM++/comp.png)

---

## Overview of the Chosen Visual Model

For this project, we used **ResNet-50**, a deep convolutional neural network developed by Microsoft Research. The architecture is well-known for its use of **residual blocks**, which make it easier to train deep networks by enabling shortcut connections that help gradients flow more effectively during backpropagation.

ResNet-50 is particularly effective for image classification tasks due to its depth and ability to capture complex features. To interpret the model’s predictions, we applied **Grad-CAM++**, a powerful explainability technique that highlights the important regions in the input image that contribute most to the model’s decision.

For generating the class activation maps, we used the **final block of layer4[-1] (Stage 4)** in the ResNet-50 architecture. This layer captures high-level semantic information, making it suitable for visualizing the reasoning behind the model's predictions.

![ResNet-50 Architecture](/Grad-CAM++/resnet50.jpg)

On the training set, we achieved an accuracy of 95.63%, while the test set accuracy was 78.5% on 10 epochs.

The classification results on the Skin Cancer MNIST: HAM10000 dataset using ResNet-50 demonstrate solid overall performance with an accuracy of 78.5%, but also highlight challenges related to class imbalance. The model performs exceptionally well on the dominant class (label 4 – likely melanocytic nevi) with an F1-score of 0.88, while underrepresented classes (e.g., labels 0, 3, and 5) show noticeably lower precision and recall. This imbalance leads to a macro average F1-score of 0.63, indicating that while the model performs well on average when weighted by class size, its ability to generalize across all classes is limited. Improving minority class performance may require techniques like data augmentation, oversampling, or fine-tuning class weights.

---

## Mini Tutorial: How to Apply Grad-CAM++ to Your Model

Here is a simplified algorithm for applying Grad-CAM++ to any CNN-based model in PyTorch:

```python
import torch
from torchvision import models, transforms
from PIL import Image
from gradcam import GradCAMPlusPlus  # Check our implementation in the Colab notebook

# Step 1: Load a pretrained model
model = models.resnet50(pretrained=True)
model.eval()

# Step 2: Set up Grad-CAM++
gradcam = GradCAMPlusPlus(model, target_layer="layer4")

# Step 3: Load and preprocess the image
image = Image.open("example.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

# Step 4: Generate the Grad-CAM++ heatmap
heatmap = gradcam(input_tensor, class_idx=None)  # Automatically uses the predicted class

# Step 5: Overlay heatmap on original image
gradcam.visualize(image, heatmap)
```

---

## Experiments of Grad-CAM++ on ResNet-50

In our project, we fine-tuned ResNet-50 using the HAM10000 dataset for skin lesion classification. We then applied Grad-CAM++ to generate class activation heatmaps to understand which parts of the image influenced the model's decision.

<div style="display: flex; align-items: center; justify-content: space-between;">

<div style="flex: 1; padding-right: 20px;">
    
The dataset used in this project is the **Skin Cancer MNIST: HAM10000**, sourced from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data). It comprises **10,015 dermatoscopic images** of pigmented skin lesions collected from multiple sources and populations, with a variety of acquisition modalities. The dataset serves as an excellent resource for training and evaluating machine learning models for skin lesion classification.

</div>

<div style="flex-shrink: 0;">
    <img src="https://s4.gifyu.com/images/bpCkE.gif" alt="GradCAM++ results GIF" style="max-height: 300px;">
</div>

</div>


The results were promising: the Grad-CAM++ visualizations often highlighted the core lesion areas, giving us confidence that the model focuses on medically relevant regions. This is particularly important in healthcare settings, where model transparency and trust are crucial.

The notebook includes examples where the original image, the activation heatmap, and the overlaid visualization are displayed side-by-side for clear comparison.

In the following GIF it can be noteced that the model predicts the label of the image and the presence or absence of cancer. 
There is also important information about visualization using hitmans.  For a simplified understanding, we have made the opposite display for images with and without a certain cancer. In the cancer images, the model's main focus and the mole itself will be in red. The images are cancer-free. the part that signals this - the birthmark - will be blue, so that the difference is immediately visually clear the difference between cancer and bening labelled images.


<div style="flex-shrink: 0;">
    <img src="https://s4.gifyu.com/images/bpCQL.gif" alt="GradCAM++ results GIF" style="max-height: 300px;">
</div>


---

## Results

The results show that all three methods—**Grad-CAM**, **Grad-CAM++**, and **CAM**—are effective at highlighting important regions in the image that the model uses for prediction. However, each has its own strengths:

- **Grad-CAM++** creates the most detailed and precise heatmaps. It’s especially good at highlighting small or fragmented areas, which is helpful when fine detail matters.
- **Grad-CAM** is slightly more general, producing smoother heatmaps, but still gives reliable and consistent results. It’s a strong all-around option.
- **CAM** doesn’t use gradients, so its heatmaps are less focused. Still, it’s useful as a second check because it confirms where the model is paying attention without relying on backpropagation.

An important insight from our experiment is that when all three methods agree on the key regions, we can be much more confident in the model’s prediction. This kind of cross-validation is especially useful in medical imaging, where being wrong can have serious consequences.

![Unknown-2](https://hackmd.io/_uploads/BkDWlkrJge.png)
![Unknown-3](https://hackmd.io/_uploads/Sk0Wl1rJxe.png)
![Unknown-4](https://hackmd.io/_uploads/SkZzgJH1ll.png)

---

## Conclusion

This project underscores the importance of explainability in deep learning, particularly within the context of medical imaging. We demonstrated how Grad-CAM++ enhances the interpretability of a ResNet-50 model trained on the HAM10000 skin cancer dataset. Compared to traditional Grad-CAM, Grad-CAM++ produced sharper and more accurate visual explanations while being computationally more efficient than CAM.

Grad-CAM++ emerges as a practical tool for researchers and practitioners seeking model transparency without significantly compromising performance. It contributes to the development of responsible AI and helps bridge the gap between black-box predictions and human interpretability.

We believe that Grad-CAM++ can serve as a reliable second opinion in clinical decision-support systems — especially valuable when AI-generated insights align with human expertise.

---

## Google Colab Link

To check our implementation: [Google Colab](https://colab.research.google.com/drive/1NlSx-AVHM8iu62ezZddtWghuCafMkXT1?usp=sharing)

---

## References

1. **Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks**  
   Aditya Chattopadhyay et al.  
   [arXiv:1710.11063](https://arxiv.org/abs/1710.11063)  
   *Original paper introducing Grad-CAM++*

2. **HAM10000 Dataset**  
   Kaggle: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)  
   *Dermatoscopic images of pigmented lesions for skin cancer classification*

3. **Grad-CAM: Visual Explanations from Deep Networks**  
   Ramprasaath R. Selvaraju et al.  
   [arXiv:1610.02391](https://arxiv.org/pdf/1610.02391)  
   *Foundational work on gradient-based class activation mapping*