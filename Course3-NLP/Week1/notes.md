# Natural Language Processing

When we were dealing with images, it was relatively easy for us to feed them into a neural network, as the pixel values were already numbers. And the network could learn parameters of functions that could be used to fit classes to labels. But what happens with text? How can we do that with sentences and words?

## Sentiment in text

### Using APIs

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat'
]

# Create an instance of the tokenizer
# In this case, using 100 which is way too big, as there are only five distinct words in this data*
tokenizer = Tokenizer(num_words = 100)
# The fit_on_texts method of the tokenizer then takes in the data and encodes it
tokenizer.fit_on_texts(sentences)
# The tokenizer provides a word index property which returns a dictionary containing key value pairs, where the key is the word, and the value is the token for that word
word_index = tokenizer.word_index
print(word_index)
```
*If you're creating a training set based on lots of text, you usually don't know how many unique distinct words there are in that text. So by setting this hyperparameter, what the tokenizer will do is take the top 100 words by volume and just encode those. It's a handy shortcut when dealing with lots of data, and worth experimenting with when you train with real data later in this course. Sometimes the impact of less words can be minimal in training accuracy, but huge in training time, but do use it carefully.



The next step will be to turn your sentences into lists of values based on these tokens:

```python
# Call on the tokenizer to get texts to sequences, and it will turn them into a set of sequences
sequences = tokenizer.text_to_sequences(sentences)

print(sentences)
```
### Output

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/output1.png "code output")

* At the top is the new dictionary
* At the bottom is my list of sentences that have been encoded into integer lists, with the tokens replacing the words


One really handy thing about this is the fact that the `texts_to_sequences` can take any set of sentences, so it can encode them based on the word set that it learned from the one that was passed into fit on texts.


If you train a neural network on a corpus of texts, and the text has a word index generated from it, then when you want to do inference with the train model, you'll have to encode the text that you want to infer on with the same word index, otherwise it would be meaningless. For example:

```python
test_data = [
    'I really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
```


In many cases, it's a good idea to instead of just ignoring unseen words, to put a special value in when an unseen word is encountered. You can do this with a property on the tokenizer:

```python
# Specify that I want the token oov for 'out-of-vocabulary' to be used for words that aren't in the word index
tokenizer = Tokenizer(num_words = 100, ovv_token = "<00V>")
```


Before you can train with texts, we needed to have some level of uniformity of size, so padding is another step:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences)
print(padded)
```
### Output

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course3-NLP/imgs/output2.png "code output 2")

```python
padded = pad_sequences(sequences, padding = 'post', truncating = 'post', maxlen = 5)
```

`padding = 'post` : padding is after the sentence and not before (as previous example)


`maxlen = 5` : set sentences to have a maximum of five words


This leads to the question: If I have sentences longer than the max length, then from where will I lose information?.


Padding default is pre, which means that you will lose from the beginning of the sentence.


If you want to override this so that you lose from the end instead, you can do so with the `truncating` parameter.

## Resources

[Text Data Preprocessing](https://keras.io/api/preprocessing/text/)
[Sarcasm in News Headlines Dataset by Rishabh Misra](https://rishabhmisra.github.io/publications/)
