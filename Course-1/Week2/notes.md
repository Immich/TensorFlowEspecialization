# An Introduction to Computer Vision

Computer vision is the field of having a computer understand and label what is present in an image.

* The smaller, the better because the computer has less processing to do.
*  It is necessary to retain enough information to be sure that the features and the object can still be distinguished.
* The images are also in gray scale, so the amount of information is also reduced.
  * Each pixel can be represented in values from zero to 255 and so it's only one byte per pixel.

### Writing code to load training data

We simply declare an object of type MNIST loading it from the Keras database. On this object, if we call the load data method, it will return four lists to us:
* The training data
* The training labels
* The testing data
* The testing labels
```python
fashion_mnist = keras.datasets.fashion_mnist(train_images, train_labels) , (test_images, test_labels) = fashion_mnist.load_data()
```

## Coding a Computer Vision Neural Network

Now we have 3 layers:
```python
model = keras.Sequential([
keras.layers.Flatten(input_shape = (28, 28)),
keras.layers.Dense(128, activation = tf.nn.relu),
keras.layers.Dense(10, activation = tf.nn.softmax)
])
```
**Sequential**: That defines a SEQUENCE of layers in the neural network


**Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.


**Dense**: Adds a layer of neurons


Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but we use these for now:

* **Relu** effectively means `If X > 0 return X, else return 0` –– so what it does it it only passes values 0 or greater to the next layer in the network.
* **Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] –– The goal is to save a lot of coding!

More details:

* The last layer has 10 neurons in it because we have ten classes of clothing in the dataset. They should always match.
* The first layer is a **flatten layer** with the input shaping 28 by 28 (remember our images are 28 by 28, so we're specifying that this is the shape that we should expect the data to be in).
* Flatten takes this 28 by 28 square and turns it into a simple linear array.
* The interesting stuff happens in the middle layer, sometimes also called a **hidden layer**.
  * This is a 128 neurons in it, and we can think about these as variables in a function. Maybe call them x1, x2 x3, etc.

## Callbacks

How can I stop training when I reach a point that I want to be at? What do I always have to hard code it to go for certain number of epochs?


The training loop supports callbacks. So in every epoch, you can callback to a code function, having checked the metrics. If they're what you want to say, then you can cancel the training at that point.

```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# First, we instantiate the class that we just created, we do that with this code.
callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Then, we used the callbacks parameter and pass it this instance of the class.
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```


More on colab attached to this week's folder.


## Resources
[Fashion - MNIST](https://github.com/zalandoresearch/fashion-mnist)


