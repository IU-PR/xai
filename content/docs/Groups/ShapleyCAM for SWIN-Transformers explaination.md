# Explaining SWIN Transformer Predictions on Traffic Sign Recognition with Shapley-CAM

## Introduction

Traffic Sign Recognition (TSR) is a vital computer vision challenge for autonomous driving systems, requiring accurate identification of signs despite varying environmental conditions and occlusions. SWIN Transformer architecture offers significant advantages for TSR through its ability to capture both local and global image dependencies via hierarchical feature representation and shifted windowing mechanisms. As these sophisticated models are increasingly deployed in safety-critical applications, Shapley-CAM provides essential interpretability by combining Shapley values with Class Activation Mapping techniques, enhancing trust and enabling more effective debugging of these complex vision systems.

## Model Overview
### SWIN Transformer

The Shifted Window (SWIN) Transformer represents a significant advancement in vision transformer architecture by introducing hierarchical representation and shifted windows. SWIN implements a hierarchical structure with varying window sizes across different layers. This design efficiently computes self-attention within local windows while allowing for cross-window connections through the window-shifting mechanism.

SWIN Transformer is particularly well-suited for traffic sign recognition for several compelling reasons:

1. **Multi-scale feature representation**: Traffic signs appear at various scales in real-world scenarios. SWIN's hierarchical design naturally captures features at different resolutions, improving recognition across varying distances and sign sizes.

2. **Context-aware processing**: By combining local attention within windows and connections between windows, SWIN can simultaneously focus on fine details of sign symbols while understanding their context within the broader scene.

3. **Robust to occlusions**: The transformer's attention mechanism helps maintain performance even when signs are partially obscured by other objects, weather conditions, or lighting variations.

### Dataset

#### Description of Dataset
This study utilizes the German Traffic Sign Recognition Benchmark (GTSRB), a well-established dataset for traffic sign classification tasks. The GTSRB contains over 50,000 images of traffic signs spread across 43 different classes, representing various traffic sign categories including speed limits, no entry, yield, and stop signs. The dataset features real-world images captured under diverse conditions, presenting several challenges:

- Varying illumination and weather conditions
- Different viewing angles and distances
- Partial occlusions and physical damage
- Imbalanced class distribution (some sign types appear more frequently than others)
- Resolution variations across samples

The dataset is divided into training and testing sets, with approximately 39,209 images for training and 12,630 images for testing. Each image in the dataset is annotated with its corresponding traffic sign class, making it suitable for supervised learning approaches.

### Fine-tune

The SWIN Transformer (`swin_tiny_patch4_window7_224`) was trained on the GTSRB dataset, leveraging data augmentation techniques such as random horizontal flips, rotations, and resized crops to improve generalization and robustness to real-world variations. The model was initialized with pretrained ImageNet weights and fine-tuned for 43 traffic sign classes using the AdamW optimizer and cross-entropy loss. Training was performed for 10 epochs with a batch size of 128, and validation accuracy was monitored after each epoch to track performance and prevent overfitting. After training, the model achieved a validation accuracy of approximately 98% and a test accuracy of around 97%, demonstrating strong recognition performance across diverse traffic sign categories.

## Explainability Method: Shapley-CAM

### What is Shapley-CAM?

Shapley-CAM is an interpretability technique that marries the Shapley values from game theory with Class Activation Mapping technique. Unlike gradient-based or activation-based CAM methods that weight feature maps by their activation strength or gradient magnitude, Shapley-CAM computes each map's exact marginal contribution to the prediction under all possible coalitions. By doing so, Shapley-CAM ensures that contributions sum to the model output (efficiency), treat identical players equally (symmetry), and assign zero value to non-influential players (dummy).

This fusion yields maps that reduce noise and highlight only those regions that truly impact the prediction. Shapley-CAM's theoretical foundation provides interpretability guarantees often absent in standard CAM variants, making it valuable for safety-critical applications.

### How it Works

Shapley-CAM frames the attribution of class predictions as a cooperative game where each feature map (in CNNs) or attention head/window (in transformers) is considered as a "player". To estimate each player's contribution, method approximates Shapley value of concrete player by sampling random subsets of other players: for each sampled coalition S, we compute the model's target-class score with S alone, then with S ∪ {i}, and record the change. Averaging these marginal contributions across many samples yields φ_i, a fair measure of how much adding feature map i shifts the prediction toward the target class.

Once the Shapley values are estimated, Shapley-CAM multiplies each feature map (or attention map) by its corresponding φ_i, sums them spatially, and applies a ReLU to focus on positive contributions. This weighted aggregation produces a single two-dimensional saliency map that highlights image regions most responsible for the model's decision, satisfying axiomatic properties (efficiency, symmetry, dummy, additivity) that traditional CAM approaches lack.

![image](/ShapleyCAM_SWIN/CAM_comparison.png)
###### *Visualization of the CAM techniques comparison for 3 classes (tiger cat, boxer, and yellow lady's slipper)*


## Applying Shapley‑CAM to SWIN Transformers

In our implementation, we leverage gradient‑based Shapley approximations to efficiently compute contribution scores for the hierarchical attention windows of the SWIN Transformer. We place forward and backward hooks on the target normalization layer (`LayerNorm`) to capture activations and gradients, then compute Hessian–vector products (HVPs) to obtain Shapley weights. The pipeline consists of four main steps:

1. **Model Loading and Preparation**  
   ‑ Instantiate the SWIN model (e.g., `swin_tiny_patch4_window7_224`) and load fine‑tuned weights from a checkpoint.  

2. **CAM Initialization**  
   ‑ Create a `ShapleyCAM` object with:  
     - The `LayerNorm` layer as the target for feature capture.  
     - A `reshape_transform` that maps windowed attention back to spatial dimensions.  
   ‑ Hooks record activations and gradients during a forward‑backward pass.

3. **Heatmap Computation**  
   ‑ **Forward pass:** compute class scores and trigger activation hooks.  
   ‑ **Backward pass:** backpropagate the target class score to collect gradients and HVPs.  
   ‑ **Weight computation:** for each window \(i\), compute  
     {{<katex display>}}\phi_i = g_i - \frac{1}{2}\mathrm{HVP}_i{{</katex>}}
     where {{<katex>}}g_i{{</katex>}} is the gradient and {{<katex>}}\mathrm{HVP}_i{{</katex>}} the Hessian–vector product.
   - **Aggregation:** multiply each attention map by {{<katex>}}\phi_i{{</katex>}}, apply ReLU, resize to input resolution, and normalize.

## Results and Visualizations

### GTSRB test images

We have calculated quantitative metrics on 12630 images from the GTSRB dataset:
- Pointing Game Accuracy: 0.9617
- Average IoU: 0.6034
- Total ADCC: 0.4683
- Total Average Drop: 0.0302
- Total Coherency: 0.8738
- Total Complexity: 0.7482
- Total Inc: 0.3152
- Total Drop Indeletion: 0.6133

**Metrics Explanation:**
- **Pointing Game Accuracy**: Measures if the point of maximum activation in the explanation falls within the ground truth object.
- **Average IoU**: Intersection over Union between the explanation heatmap and ground truth regions.
- **Total ADCC**: Average Drop in Confidence when Critical pixels identified by Shapley-CAM are removed.
- **Total Average Drop**: The minimal drop in confidence when random pixels are removed validates that our explanation correctly focuses on impactful regions.
- **Total Coherency**: Measures consistency of explanations across different inputs.
- **Total Complexity**: Evaluates explanation detail level.
- **Total Inc**: Measures confidence increase when only pixels identified as important are kept.
- **Total Drop Indeletion (0.6133)**: Confidence drop when pixels marked unimportant are removed.

Below you can see several examples from the test dataset: 

<br>
<div style="display: flex; gap: 0;">
  <img src="/ShapleyCAM_SWIN/ex1.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/ex2.png" style="height:150px; margin:0; padding:0; border:none;"/>
  <img src="/ShapleyCAM_SWIN/ex3.png" style="height:150px; margin:0; padding:0; border:none;"/>
  <img src="/ShapleyCAM_SWIN/ex4.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div> 
<br>

### Ambiguous cases

We have also applied Shapley‑CAM to representative traffic sign images from the images found in the internet. The following figures show the reference image of the class that we want to predict alongside its Shapley‑CAM generated heatmap:

**Figure 1**

The top heatmap shows the Swin Transformer's attention when classifying the "70" speed limit sign—focusing mainly on the circular sign. The bottom-right heatmap is the Shapley-CAM result for the bend road warning sign, revealing that the model attends more to the triangular warning sign.

<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/70.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/1_70.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div> 

<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/bend.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/1_bend.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div>
<br>

**Figure 2**

The top-right heatmap corresponds to the "30" speed limit sign, while the bottom-right Shapley-CAM heatmap corresponds to the road work sign. The attention is nicely distributed—focused on the relevant parts of each sign.

<br>
<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/30.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/2_30.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div> 

<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/worker.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/2_work.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div>
<br>

**Figure 3**

The top-right heatmap primarily highlights the 30 speed limit sign, which aligns with the intended target. However, there's also attention spillover onto the no-overtaking sign below, which suggests the model isn't perfectly isolating the relevant region. Nevertheless, in the bottom-right heatmap, the focus is mostly on the no-overtaking sign.

<br>
<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/30.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/3_30.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div> 

<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/cars.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/3_cars.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div>
<br>

**Figure 4**

In the heatmap, the model appears to pay significant attention to the person in the warning sign. This suggests the model effectively identifies the pattern.

<br>
<div style="display: flex; gap: 100px;">
  <img src="/ShapleyCAM_SWIN/worker.png" style="height:150px; width:150px; border:none;"/>
  <img src="/ShapleyCAM_SWIN/4_toilet.png" style="height:150px; margin:0; padding:0; border:none;"/>
</div> 
<br>


## References
- [CAMs as Shapley Value-based Explainers](https://arxiv.org/pdf/2501.06261v1)
- [SWIN Transformer](https://arxiv.org/pdf/2103.14030)
- [Traffic Sign dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data)
