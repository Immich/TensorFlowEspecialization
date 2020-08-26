# WEEK 3 QUIZ

#### 1. If X is the standard notation for the input to an RNN, what are the standard notations for the outputs?
R = Y(hat) and hat

#### 2. What is a sequence to vector if an RNN has 30 cells numbered 0 to 29
R = The Y(hat) for the last cells

#### 3. What does a Lambda layer in a neural network do?
R = Allows you to execute arbitrary code while training

#### 4. What does the axis parameter of tf.expand_dims do?
R = Defines the dimension index at which you will expand the shape of the tensor

#### 5. A new loss function was introduced in this module, named after a famous statistician. What is it called?
R = Huber loss

#### 6. What’s the primary difference between a simple RNN and an LSTM
R = In addition to the H outputs, LSTMs have a cell state that runs accross all cells

#### 7. If you want to clear out all temporary variables that tensorflow might have from previous sessions, what code do you run?
R = `tf.keras.backend.clear_session()`

#### 8. What happens if you define a neural network with these two layers?
```python
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Dense(1),
```

R = Your model will fail because you need `return_sequences = True` after the first LSTM layer