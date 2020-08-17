# WEEK 3 QUIZ :computer:

#### 1. Why does sequence make a large difference when determining semantics of language?
R = Because the order in which words appear dictate their impact on the meaning of the sentence

#### 2. How do Recurrent Neural Networks help you understand the impact of sequence on meaning?
R = They carry meaning from one cell to the next

#### 3. How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?
R = Values from earlier words can be carried to later ones via a cell state

#### 4. What keras layer type allows LSTMs to look forward and backward in a sentence?
R = Bidirectional

#### 5. What’s the output shape of a bidirectional LSTM layer with 64 units?
R = (None, 128)

#### 6. When stacking LSTMs, how do you instruct an LSTM to feed the next one in the sequence?
R = Ensure that return_sequences is set True only on units that feed another LSTM

#### 7. If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?
R = (None, 116, 128)

#### 8. What’s the best way to avoid overfitting in NLP datasets?
R = It depends on the problem