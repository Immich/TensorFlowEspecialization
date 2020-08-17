# Natural Language Processing

## Sequence Models

### Recurrent Neural Networks (RNN)

The basic idea of a recurrent neural network or RNN, is often drawn a little like this:

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/rnn.png "RNN")

We have x as in input and y as an output. But there's also an element that's fed into the function from a previous function:

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/rnns.png "RNNs")

x_0 is fed into the function returning y_0. An output from the function is then fed into the next function, which gets fed into the function along with x_2 to get y_2, producing an output and continuing the sequence.


An update to RNNs is called LSTM (long short - term memory) has been created. In addition to the context being passed as it is in RNNs, LSTMs have an additional pipeline of contexts called cell state and this can pass through the network to impact it. This helps keep context from earlier tokens relevant in later ones.

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/lstm.png "LSTM")

Cell states can also be bidirectional, so later contexts can impact earlier ones.

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/bilstm.png "Bi directional LSTM")


## Implenenting LSTMs in code

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    # The parameter passed in is the number of outputs that I desire from that layer, in this case it's 64
    # If wrap that with tf.keras.layers.Bidirectional, it will make the cell state go in both directions
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```

#### Summary model
![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/modelsummary1.png "lstm code")

You can also stack LSTMs like any other keras layer by using code like this:

#### Summary model
![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/modelsummary2.png "stacked lstm code")

When you feed an LSTM into another one, you do have to put the return sequences equal true parameter into the first one. This ensures that the outputs of the LSTM match the desired inputs of the next one.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```

## Using a Convolutional Network

Another type of layer that you can use is a convolution, you specify the number of convolutions to learn, their size, and their activation function.


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_lenght = max_lenght),
    # Words will be grouped into the size of the filter, in this case, 5
    # We have 128 filters each for 5 words
    tf.keras.layers.Conv1D(128, 5, activation = 'relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```
As the size of the input was 120 words, and a filter that is 5 words long will shave off 2 words from the front and back, leaving us with 116. The 128 filters specified will show up here as part of the convolutional layer.

#### Summary model
![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/convolutionallayer.png "convolutional layer")

