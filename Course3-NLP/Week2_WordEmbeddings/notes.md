# Natural Language Processing

## Word Embeddings

This process is called embedding, Words and associated words are clustered as vectors in a multi-dimensional space.


```python
import tensorflow_datasets as tfds
import numpy as np

imbd, info = tfds.load("imdb_reviews", with_info = True, as_supervised = True)
# The data is split into 25,000 samples for training and 25,000 samples for testing
train_data, test_data = imdb['train'], imdb['test']

# Each of these are iterables containing the 25,000 respective sentences and labels as tensors
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# Iterate over training data extracting the sentences and the labels. The values for S and I are tensors, by calling their NumPy method, it extracts their value
for s, l in train_data:
    training_sentencess.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

# When training, labels are expected to be NumPy arrays
# Turn the list of labels created into NumPy arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Next, tokenize our sentences
vocab_size = 10000
embedding_dim = 16
max_lenght = 120
trunc_type = 'post'
oov_tok = <00V>

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, ovv_token = ovv_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen = max_lenght, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sequences)
testing_padded = pad_sequences(testing_sequences, max_len = max_lenght)
```

Now define the model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_lenght = max_lenght),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```

###### Output

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/flattenoutput.png "output with Flatten()")

Often in natural language processing, a different layer type than a flatten is used, and this is a global average pooling 1D:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_lenght = max_lenght),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```
The reason for this is the size of the output vector being fed into the dance:

###### Output

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/GlobalAvgPooling1Doutput.png "output with GlobalAveragePooling1D()")


Over 10 epochs with global average pooling, it got an accuracy of 0.9664 on training and 0.8187 on test, taking about 6.2 seconds per epoch. With flatten, accuracy was 1.0 and validation about 0.83 taking about 6.5 seconds per epoch. So it was a little slower, but a bit more accurate. 

### Wrap up

Not only do the meanings of the words matter, but also the sequence in which they are found.

## Resources

* [Embedding Projector](https://projector.tensorflow.org/)
* [IMDB reviews dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
* [TensorFlow Datasets](https://github.com/tensorflow/datasets/tree/master/docs/catalog)
* [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets/catalog/overview)
* [Subwords text encoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder)
* [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf)
* [3 subword algorithms help to improve your NLP model performance](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46)
