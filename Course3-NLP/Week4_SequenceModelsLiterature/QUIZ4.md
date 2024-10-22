# WEEK 4 QUIZ

#### 1. What is the name of the method used to tokenize a list of sentences?
R = fit_on_texts(sentence)

#### 2. If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?
R = (None, 116, 128)

#### 3. What is the purpose of the embedding dimension?
R = Is the number of dimensions representing the word encoding

#### 4. IMDB Reviews are either positive or negative. What type of loss function should be used in this scenario?
R = Binary crossentropy

#### 5. If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?
R = Use the pad_sequences object from the tensorflow.keras.preprocessing.sequence namespace

#### 6. When predicting words to generate poetry, the more words predicted the more likely it will end up gibberish. Why?
R = Because the probability that each word matches an existing phrase goes down the more words you create

#### 7. What is a major drawback of word-based training for text generation instead of character-based generation?
R = Because there are far more words in a typical corpus than characters, it is much more memory intensive

#### 8. How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?
R = Values from earlier words can be carried to later ones via a cell state