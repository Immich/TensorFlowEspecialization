# Natural Language Processing

## Sequence Models and Literature

### Finding what the next word should be

```python
model = Sequential()
# Define an embedding layer
# We want it to handle all of the words, so we set that in the first parameter
# The second parameter is the number of dimensions to use to plot the vector for a word
# The size of the input dimensions will be fed in, and this is the length of the longest sequence minus 1
# We subtract 1 because we cropped off the last word of each sequence to get the label, so our sequences will be 1 less than the maximum sequence length
model.add(Embedding(total_words, 64, input_lenght = max_sequence_len - 1))
# Add an LSTM, specify 20 units
model.add((LSTM(20)))
# Add a dense layer sized as the total words, which is the same size that we used for the one-hot encoding
# This layer will have one neuron per word and that neuron should light up when we predict a given word
model.add(Dense(total_words, activation = 'softmax'))
# Since the task is a categorical classification, we'll set the laws to be categorical cross entropy
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Train for about 500 epochs, as it takes a while for a model like this to converge, particularly as it has very little data
model.fit(xs, ys, epochs = 500, verbose = 1)
```

We can also set the LSTM to be Bidirectional:

```python
model.add(Bidirectional(LSTM(20)))
```

## Resources

[Text generation with an RNN](https://www.tensorflow.org/tutorials/text/text_generation)