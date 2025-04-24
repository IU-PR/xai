This work is made by **Andrei Markov** and **Nikita Bogdankov**

# Grad-CAM

## What is it?

 **Grad-CAM** (Gradient-weighted Class Activation Mapping) is a technique used in deep learning, particularly with convolutional neural networks (CNNs), to understand which regions of an input image are important for the network's prediction of a particular class.

![image](/GradCAM/example4.png)

This method can be used for understanding how a CNN has been driven to make a final classification decision. Grad-CAM is class-specific, which means that it can produce a separate visualizations on the image for each class. If happens that there is a classification error, than Grad-CAM can help you to see what it did wrong, so we can say that it makes the algorithm more transparent to the developers. 

Grad-CAM consists of producing heatmaps which show the activation classes on the input images. Each activation class is associated with one of the output classes, which are used to indicate the importance of each pixel to the question by changing the intensity of the pixel. 

![image](/GradCAM/example6.png)

On the picture above you can see how important objects for detection are highlighted on the image. And after we created heatmap, we should overlay image and this heatmap to get the final result. 

## How does it work?

Now let's see get to how Grad-CAM works. I will give you a step-by-step explanation:

1. **Forward Pass** - First, let {{<katex>}}A^{k}{{</katex>}} be the activation map. It is passed through the subsequent layers of the neural network to compute the final class score {{<katex>}}y_{c}{{</katex>}} for the target class {{<katex>}}c{{</katex>}}. {{<katex display>}}y_{c}=\sum_iw_{i}^c\cdot A_i{{</katex>}} where {{<katex>}}w_{i}^c{{</katex>}} represents the weight of the {{<katex>}}{i}{{</katex>}} th feature map {{<katex>}}A_i{{</katex>}} for the class {{<katex>}}c{{</katex>}}.  
2. **Backpropagation** -  The gradients of the class score {{<katex>}}y_c{{</katex>}} with respect to the activation map {{<katex>}}A_k{{</katex>}} are computed using backpropagation: {{<katex display>}}\dfrac{\partial y_c}{\partial A^{k}}{{</katex>}}
3. **Gradient Weighting** - Grad-CAM assigns importance weights to each activation map based on the gradients gained in the backpropagation step. This is done by taking Global Average Pooling of the gradients: {{<katex display>}}\alpha_{k}^{c}= \dfrac{1}{Z} \sum_{i} \sum_{j} \dfrac {\partial y_c}{\partial A_{ij}^{k}}{{</katex>}} {{<katex>}}\alpha_{k}^{c}{{</katex>}} is the importance weight and {{<katex>}}Z{{</katex>}} is the spatial dimension of the activation map. {{<katex>}}i{{</katex>}} and {{<katex>}}j{{</katex>}} are width and height dimensions, {{<katex>}}y_c{{</katex>}} is class score(before softmax).
4. **Heatmap Generation** - Grad-CAM generates the heatmap {{<katex>}}H_c{{</katex>}} by linearly combining the activation maps weighted by their importance scores: {{<katex display>}}H^{c} =ReLU (\sum_k \alpha_{k}^{c} A^{k}){{</katex>}} where {{<katex>}}ReLU{{</katex>}} is the rectified linear unit activation function, allowing only positive values contribute to the heatmap.
Heatmap {{<katex>}}H_c{{</katex>}} highlights the regions in the input image that are most relevant for the network's decision for the target class {{<katex>}}c{{</katex>}}. Brighter the region, more it contributes to the prediction.
![image](/GradCAM/example.png)

To clarify the process, I will attach the image explaining the described above:
![image](/GradCAM/example2.png)


## Where can it be used?

But now let's talk about where Grad-CAM can be used. There are many fields, where it can be useful, so here are some examples:

- **Medical Image Analysis** - Grad-CAM can help clinicians to quicker check the patients to see if there are any problems. If so, it shows where the problems may be located. 
![image](/GradCAM/example3.png)

- **Autonomous Vehicles** - Grad-CAM is used for object detection, lane detection, and pedestrian recognition. It can be used to visualize which parts of the input image are most relevant in the decision-making process of these systems. 
![image](/GradCAM/example5.png)
- **Security and Surveillance**: Grad-CAM is used for activity recognition, intruder detection, and object tracking. It can help security personnel to notice the criminal action if they missed it for some reason.
- **Industrial Quality Control** - Grad-CAM is used for defect detection, product classification, and quality assessment. Grad-CAM can help operators to faster detect problem details. 
- **Remote Sensing and Earth Observation** - Grad-CAM is used for land cover classification, crop monitoring, and disaster detection. Grad-CAM can assist analysts in understanding which features in satellite or aerial imagery are indicative of certain land cover types or environmental conditions, aiding in resource management and disaster response.


[Link to collab](https://colab.research.google.com/drive/13_q6BGlqTJb8gSqpUwhec2JoMhrmqLgU) 