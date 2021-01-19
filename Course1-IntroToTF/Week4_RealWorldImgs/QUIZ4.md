# WEEK 4 QUIZ :computer:

#### 1. Using Image Generator, how do you label images?
R = It's based on the directory the image is contained in

#### 2. What method on the Image Generator is used to normalize the image?
R = rescale

#### 3. How did we specify the training size for the images?
R = The target_size parameteron the training generator

#### 4. When we specify the input_shape to be (300, 300, 3), what does that mean?
R = Every image will be 300x300 pixels, with 3 bytes to define color

#### 5. If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here?
R = You're overfitting on your training data

#### 6. Convolutional Neural Networks are better for classifying images like horses and humans because:
* In these images, the features may be in different parts of the frame
* There's a wide variety of horses
* There's a wide variety of humans

#### 7. After reducing the size of the images, the training results were different. Why?
R = We removed some convolutions to handle the smaller images
