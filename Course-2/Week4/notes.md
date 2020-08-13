# Convolutional Neural Networks

## Multiclass Classification 

Replicate this for multiple classes:
![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-2/imgs/multipleclassimagegen.png "Multiple Classes Image Generator")


Instead of 'binary', we should change class_mode to 'categorical':

```python
train_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size = (300, 300),
                batch_size = 128,
                class_mode = 'categorical')
```

Another change to do from previous code is in the output layer:

```python
tf.keras.layers.Dense(3, activation = 'softmax')
```
Now we have an output layer that has three neurons, one for each of the classes rock, paper, and scissors, and it's activated by softmax which turns all the values into probabilities that will sum up to one.


The output of a neural network with three neurons and a softmax would reflect that, and maybe look like this with a very low probability of rock, a really high one for paper, and a decent one for scissors. All three probabilities would still add up to one.

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-2/imgs/softmax.png "softmax explication")


The final change comes when compiling:

```python
model.compile(loss = 'categorical_crossentropy',
            optimizer = RMSprop(lr = 0.0001),
            metrics = ['acc'])
```


## Resources

* [Rock-Paper-Scissors dataset](http://www.laurencemoroney.com/rock-paper-scissors-dataset/)
* [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) (This was used for Final Assignment)