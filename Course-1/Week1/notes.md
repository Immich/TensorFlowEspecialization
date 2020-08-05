# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
## A new program paradigm

![alt text]( "Programming paradigms")

## The "Hello World" of Machine Learning

Here's our first line of code:

```python
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape[1])])
```
This is written using Python and TensorFlow and an API in TensorFlow called keras.


Keras makes it really easy to define neural networks. **A neural network is basically a set of functions which can learn patterns**.

* In keras, you use the word `Dense` to define a layer of connected neurons. There's only one dense here.
* So there's only one layer and there's only one unit in it, so it's a single neuron.
* Successive layers are defined in sequence, hence the word `Sequential`.
* Here's only one. So we have a single neuron.
* We define the shape of what's input to the neural network in the first and in this case the only layer, and we can see that our input shape is super simple. It's just one value.


There are two function roles that you should be aware of though and these are loss functions and optimizers. This code defines them:
```python
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
```

In this case, the loss is mean squared error and the optimizer is SGD which stands for stochastic gradient descent.

* The neural network has no idea of the relationship between X and Y, so it makes a guess. Say it guesses Y equals 10X minus 10.
* It will then use the data that it knows about, that's the set of Xs and Ys that we've already seen to measure how good or how bad its guess was.
* The `loss function` measures this and then gives the data to the `optimizer` which figures out the next guess.
* So the `optimizer` thinks about how good or how badly the guess was done using the data from the `loss function`.
* Then the logic is that each guess should be better than the one before.
* As the guesses get better and better, an `accuracy` approaches 100 percent, the term `convergence` is used.


Our next step is to represent the known data.

```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)
```
The np.array is using a Python library called numpy that makes data representation particularly enlists much easier.



The training takes place in the fit command.
```python
model.fit(xs, ys, epochs = 500)
```
* Here we're asking the model to figure out how to fit the X values to the Y values.
* The epochs equals 500 value means that it will go through the training loop 500 times.
* This training loop is what we described earlier:
  * Make a guess, measure how good or how bad the guesses with the loss function, then use the optimizer and the data to make another guess and repeat this.

When the model has finished training, it will then give you back values using the predict method.
```python
print(model.predict([10.0]))
```
#### So it hasn't previously seen 10, and what do you think it will return when you pass it a 10?

Now you might think it would return 19 because after all Y equals 2X minus 1, and you think it should be 19. But when you try this in the workbook yourself, you'll see that it will return a value very close to 19 but not exactly 19.

#### Now why do you think that would :
1. You trained it using very little data. There's only six points. Those six points are linear but there's no guarantee that for every X, the relationship will be Y equals 2X minus 1.
2. There's a very high probability that Y equals 19 for X equals 10, but the neural network isn't positive. So it will figure out a realistic value for Y.


When using neural networks, as they try to figure out the answers for everything, they deal in probability. You'll see that a lot and you'll have to adjust how you handle answers to fit.







