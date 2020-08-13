# WEEK 4 QUIZ :computer:

1. The diagram for traditional programming had Rules and Data In, but what came out?
R = Answers

2. Why does the DNN for Fashion MNIST have 10 output neurons?
R = The dataset has 10 classes

3. What is a Convolution?
R = A technique to extract features from an image

4. pplying Convolutions on top of a DNN will have what impact on training?
R = It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN.

5. What method on an ImageGenerator is used to normalize the image?
R = rescale

6. When using Image Augmentation with the ImageDataGenerator, what happens to your raw image data on-disk.
R = Nothing

7. Can you use Image augmentation with Transfer Learning?
R = Yes, it's pre-trained layers that are frozen. So you can augment your images as you train the bottom layers of the DNN with them.

8. When training for multiple classes what is the Class Mode for Image Augmentation?
R = class_mode = 'categorical'