# Style-Transfer

To use the style transfer, enter `python StyleTransferUser.py` and select the desired style and content image. Note that the image has to be square, jpg format.



## Prerequisite

Python 3+

Pytorch

Matplotlib

TorchVision



# CSE 455 Final Project - Style Transfer

Group members: Andrew Shi,
Alex Jiang,
Xin Lin netid: xlin7799,
Xiaxuan Gao netid: gaox26

Video Link: https://youtu.be/em0WcErCxFk

## Background

In the final project, we built a CNN-Based style transformer for images. Given a content image and a style image, our tool should be able to produce a mixed image that retain all of the edges and contours of content image but have the colors and textures of the style image.

## Method

We used the pretrained convolutional neural network VGG-19 as a feature extractor to make sure that the mixed output image preserves some features from both the content image and the style image. The high level overview of our image style transfer pipeline is that we start with a noisy input image and then compute the loss, which is the mean squared error(MSE) difference between the feature maps of the input image and both the content and style image. By taking the derivative of the loss function with respect to the input image at each iteration, the input image is updated through gradient descent and eventually resembles both the content and style images. The main challenge in our project is to correctly compute the loss function.

### Content Loss

We feed both the content image and the noisy input image to the VGG-19 network. VGG-19 has 19 convolutional layers. The output at each of the convolutional layer can be used as the feature map. At one or more of the convolutional layers, we compute the MSE difference between that of content image and noisy input image and use the combined difference as the content loss. 

### Style Loss

Computing style loss is more tricky. Instead of having exactly the same feature activation at each of the convolutional layers, our objective here is to ensure that features that are activated in the same layer for content image are also activated together in the same layer for the noisy input image. Therefore, for each of the convolutional layer, we compute the dot product of the output with its transpose. Entries that have high value implies that the two features are activated together. The dot product is called the gram matrix. We compute the gram matrix for both the style image and input image at four convolutional layers, and then derive the style loss by calculating the MSE difference between gram matrices at each layer.

## Result
Given the original image and the style image.
![content](/data/content.jpg)
![content](/data/style.jpg)

\
We can get the transformed image like below.
![Result](/data/Result.png)

## Discussion
The most difficult problem we encountered was understanding the math proof behind this neural transform. We spent quit a lot of time working on coding the style loss as well as feature extraction. The next step for us is to make this style transform happening in the live video. And we expect it to work like the instagram filter. The main difference between our work and other people's work is that we used VGG-19 pretrained network. This enables us to not to train a neural network from scratch.