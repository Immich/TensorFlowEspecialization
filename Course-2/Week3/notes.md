# Convolutional Neural Networks

## Transfer Learning :brain:

Rather than needing to train a neural network from scratch we can need a lot of data and take a long time to train, you can instead download maybe an open-source model that someone else has already trained on a huge dataset maybe for weeks and use those parameters as a starting point to then train your model just a little bit more on perhaps a smaller dataset that you have for a given task, so it is called transfer learning.

### Coding transfer learning from the inception mode

```python
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# The inception V3 has a fully-connected layer at the top
# By setting include_top to false, we specify that we want to ignore this and get straight to the convolutions
pre_trained_model = InceptionV3(input_shape(150, 150, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Now that we have the pretrained model instantiated, iterate through its layers and lock them, saying that they're not going to be trainable with this code
for layer in pre_trained_model.layers:
    layer.trainable = False

# Moved up the model description to find mixed7, which is the output of a lot of convolution that are 7 by 7
# With this code, I'm going to grab that layer from inception and take it to output
last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output

# Define our new model, taking the output from the inception model's mixed7 layer, which we called last_ouput
from tensorflow.keras.optimizers import RMSprop

# You start by flattening the input, which just happens to be the output from inception
x = layers.Flatten()(last output)
# Add a Dense hidden layer
x = layers.Dense(1024, activation = 'relu')(x)
# Add output layer which has just one neuron activated by a sigmoid to classify between two items
x = layers.Dense(1, activation = 'sigmoid')(x)

# Create a model using the Model abstract class
# Passing at the input and the layers definition that we just created
model = Model(pretrained_model.input, x)
model.compile(optimizer = RMSprop(lr = 0.0001),
            loss = 'binary_crossentropy',
            metrics = ['acc'])

```

There's another layer take in Keras called a `dropout`. And the idea behind the dropout is that layers in a neural network can sometimes end up having similar weights and possible impact each other leading to over-fitting.


Dense layers can look a little bit like this:
![alt-text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-2/imgs/denselayer.png "Dense Layer")

By dropping some out, we make it look like this:
![alt-text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-2/imgs/dropoutlayer.png "Layers Droped out")
And that has the effect of neighbors not affecting each other too much and potentially removing overfitting.


Adding dropout to code:
```python
# The parameter is between 0 and 1 and it's the fraction of units to drop
# In this case, we're dropping out 20% of our neurons
x = layers.Dropout(0.2)(x)
```


A copy of the pretrained weights for the inception neural network is saved at this [URL](https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels). Think of this as a snapshot of the model after being trained. It's the parameters that can then get loaded into the skeleton of the model, to turn it back into a trained model.

## Resources

* [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
* [ImageNet](http://www.image-net.org/)
* [Understanding Dropout](https://www.youtube.com/watch?v=ARq74QuavAo)