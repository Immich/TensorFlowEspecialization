# Enhancing Vision with Convolutional Neural Networks

One of the things that you would have seen when you looked at the images is that there's a lot of wasted space in each image. While there are only 784 pixels, it will be interesting to see if there was a way that we could condense the image down to the important features that distinguish what makes it a shoe, or a handbag, or a shirt. That's where convolutions come in.

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-1/imgs/cnn.png "Convolutional Neural Nets")

For every pixel, take its value, and take a look at the value of its neighbors. If our filter is three by three, then we can take a look at the immediate neighbor, so that you have a corresponding three by three grid. Then to get the new value for the pixel, we simply multiply each neighbor by the corresponding value in the filter.


The idea here is that some convolutions will change the image in such a way that certain features in the image get emphasized.

## Pooling

Pooling is a way of compressing an image. A quick and easy way to do this, is to go over the image of four pixels at a time, i.e, the current pixel and its neighbors underneath and to the right of it. Of these four, pick the biggest value and keep just that.

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-1/imgs/pooling.png "Pooling")


```python
model = tf.keras.Sequential([
  # Line 1
  tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
  # Line 2
  tf.keras.layers.MaxPooling2D(2, 2),
  # Line 3
  tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
  # Line 4
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dense(10, activation = 'softmax'),
])
```
###### Line 1 
Here we're asking keras to generate 64 filters for us. These filters are 3 by 3, their activation is relu, which means the negative values will be thrown way, and finally the input shape is as before, the 28 by 28. That extra 1 just means that we are tallying using a single byte for color depth. As we saw before our image is our gray scale, so we just use one byte.

###### Line 2
This line of code will then create a pooling layer. It's max-pooling because we're going to take the maximum value. We're saying it's a two-by-two pool, so for every four pixels, the biggest one will survive as shown earlier.

###### Line 3 & 4
We then add another convolutional layer, and another max-pooling layer so that the network can learn another set of convolutions on top of the existing one, and then again, pool to reduce the size.



A really useful method on the model is the `model.summary()` method. This allows you to inspect the layers of the model, and see the journey of the image through the convolutions.








