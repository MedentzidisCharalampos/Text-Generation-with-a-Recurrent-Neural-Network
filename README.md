# Text-Generation-with-a-Recurrent-Neural-Network
Text Genetation using characted based RNN.  We use a dataset of Shakespeare's writing from Andrej Karpathy's. Given a sequence of characters from this data ("Shakespear"), a model is trained to predict the next character in the sequence. Longer sequences of text can be generated by calling the model repeatedly.

# Output of the model
The following is sample output when the model in this tutorial trained for 30 epochs, and started with the string "Q":

QUEENE:
I had thought thou hadst a Roman; for the oracle,
Thus by All bids the man against the word,
Which are so weak of care, by old care done;
Your children were in your holy love,
And the precipitation through the bleeding throne.

BISHOP OF ELY:
Marry, and will, my lord, to weep in such a one were prettiest;
Yet now I was adopted heir
Of the world's lamentable day,
To watch the next way with his father with his face?

ESCALUS:
The cause why then we are all resolved more sons.

VOLUMNIA:
O, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, it is no sin it should be dead,
And love and pale as any will to that word.

QUEEN ELIZABETH:
But how long have I heard the soul for this world,
And show his hands of life be proved to stand.

PETRUCHIO:
I say he look'd on, if I must be content
To stay him from the fatal of our country's bliss.
His lordship pluck'd from this sentence then for prey,
And then let us twain, being the moon,
were she such a case as fills m

1. While some of the sentences are grammatical, most do not make sense. The model has not learned the meaning of words, but consider:

2. The model is character-based. When training started, the model did not know how to spell an English word, or that words were even a unit of text.

3. The structure of the output resembles a play—blocks of text generally begin with a speaker name, in all capital letters similar to the dataset.
T
4. As demonstrated below, the model is trained on small batches of text (100 characters each), and is still able to generate a longer sequence of text with coherent structure.

# Process the Text
1. Vectorize the text:   
  Before training, we need to map strings to a numerical representation. Create two lookup tables: one mapping characters to numbers, and another for numbers to characters.
2. Create training examples and targets:  
  Next divide the text into example sequences. 
  Each input sequence will contain seq_length characters from the text.
  For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.
  So break the text into chunks of seq_length+1. 
  For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".
  
3. Create training batches:  
  Before feeding this data into the model, we need to shuffle the data and pack it into batches.

# Build The Model
Use tf.keras.Sequential to define the model. 

1. tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map the numbers of each character to a vector with embedding_dim dimensions;
2. tf.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use an LSTM layer here.)
3. tf.keras.layers.Dense: The output layer, with vocab_size outputs.

For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input, and applies the dense layer to generate logits predicting the log-likelihood of the next character:

![alt text](https://github.com/MedentzidisCharalampos/Text-Generation-with-a-Recurrent-Neural-Network/blob/main/model_architecture.png)

# Train the model

The problem can be treated as a standard classification problem.     
Given the previous RNN state, and the input this time step, predict the class of the next character.

Attach an optimizer, and a loss function: 

The standard tf.keras.losses.sparse_categorical_crossentropy loss function works in this case because it is applied across the last dimension of the predictions.    
Configure the training procedure using the tf.keras.Model.compile method.  
Use tf.keras.optimizers.Adam with default arguments and the loss function.

Configure checkpoints:

Use a tf.keras.callbacks.ModelCheckpoint to ensure that checkpoints are saved during training.

Execute the training:

To keep training time reasonable, use 10 epochs to train the model.

# Generate text
Restore the latest checkpoint:  

1. To keep this prediction step simple, use a batch size of 1.

2. Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.

3. To run the model with a different batch_size, you need to rebuild the model and restore the weights from the checkpoint.

# The prediction loop

The following code block generates the text:

1. Begin by choosing a start string, initializing the RNN state and setting the number of characters to generate.

2. Get the prediction distribution of the next character using the start string and the RNN state.

3. Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.

4. The RNN state returned by the model is fed back into the model so that it now has more context, instead of only one character. After predicting the next character, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted characters.

![alt text](https://github.com/MedentzidisCharalampos/Text-Generation-with-a-Recurrent-Neural-Network/blob/main/prediction_loop.png)
